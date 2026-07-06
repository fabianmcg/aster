// Kernel: C[i, j] = C[i, j] * rsqrt(inv_d * sum(D[i, :]) + eps) for all rows i and columns j.
//
// C is column-major bf16 (2 bytes per element), i.e. C[i, j] lives at element
// offset j * M + i.  D is row-major f32 (4 bytes per element), i.e. D[i, k]
// lives at element offset i * n_d + k.  Constraint: n_d <= 64.
//
// Grid: bid.x selects a 256-row band; bid.y selects a 256-column tile.
// Block: multiple of 64 threads (256 threads = 4 wavefronts at launch time).
//
// Algorithm:
//   Phase 1 (all waves): each wavefront reduces 64 consecutive rows.  For
//   each row r in [0, 64), each lane loads D[wavefrontRowBase + r, laneId]
//   (or 0.0 when laneId >= n_d or row >= m), then a 6-round butterfly
//   reduces the 64 lanes so every lane holds the full row sum.  All lanes
//   then compute rsqrt(inv_d * sum + eps) (the reciprocal square root) and
//   write it to LDS at (waveId * 64 + r) * 4; concurrent same-address writes
//   are safe because all lanes carry the same value after the butterfly.
//   Phase 2: s_barrier so all waves see the LDS scales.
//   Phase 3 (all threads): each thread owns globalRow = bid.x * 256 + tid.
//   It reads its scale from LDS[tid * 4] and, for each column j in
//   [bid.y * 256, min(bid.y * 256 + 256, n)), multiplies C[globalRow, j] by it.
//
// Shared memory: 256 rows * 4 bytes = 1024 bytes.

#map_lane   = affine_map<(d0) -> (d0 mod 64)>
#map_wid    = affine_map<(d0) -> (d0 floordiv 64)>
#map_elem   = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
#map_times2 = affine_map<(d0) -> (d0 * 2)>
#map_times4 = affine_map<(d0) -> (d0 * 4)>
#map_add    = affine_map<(d0, d1) -> (d0 + d1)>

amdgcn.module @colv2_mod target = #amdgcn.target<gfx942> {
  func.func @colv2(
      %c_ptr : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %d_ptr : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %m     : i32,
      %n     : i32,
      %n_d   : i32,
      %inv_d : f32,
      %eps   : f32)
      attributes {gpu.kernel, gpu.shared_memory_size = 1024 : i32} {

    %c0_i32  = arith.constant 0    : i32
    %c2_i32  = arith.constant 2    : i32
    %cm1_i32 = arith.constant -1   : i32
    %c0f     = arith.constant 0.0  : f32

    %c0_idx   = arith.constant 0   : index
    %c1_idx   = arith.constant 1   : index
    %c6_idx   = arith.constant 6   : index
    %c64_idx  = arith.constant 64  : index
    %c256_idx = arith.constant 256 : index

    %tid  = gpu.thread_id x
    %bidx = gpu.block_id  x
    %bidy = gpu.block_id  y

    %m_idx   = arith.index_cast %m   : i32 to index
    %n_idx   = arith.index_cast %n   : i32 to index
    %n_d_idx = arith.index_cast %n_d : i32 to index

    %lane_idx = affine.apply #map_lane(%tid)
    %wid_idx  = affine.apply #map_wid(%tid)
    %lane_i32 = arith.index_cast %lane_idx : index to i32

    // -------------------------------------------------------------------------
    // Phase 1: all waves accumulate D row sums and write scales to LDS.
    // Each wave covers 64 rows: wavefrontRowBase = bid.x * 256 + wid * 64.
    // -------------------------------------------------------------------------

    // Buffer descriptor for D: num_records = m * n_d * 4 bytes, stride = 0.
    %total_elems   = affine.apply #map_elem(%m_idx, %c0_idx)[%n_d_idx]
    %total_bytes_i = affine.apply #map_times4(%total_elems)
    %total_bytes   = arith.index_cast %total_bytes_i : index to i32
    %d_sgpr        = lsir.to_reg %d_ptr      : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> !amdgcn.sgpr<[? + 2]>
    %total_bytes_s = lsir.to_reg %total_bytes : i32 -> !amdgcn.sgpr
    %c0_s          = lsir.to_reg %c0_i32     : i32 -> !amdgcn.sgpr
    %d_rsrc = amdgcn.make_buffer_rsrc %d_sgpr, %total_bytes_s, %c0_i32,
        cache_swizzle = false, swizzle_enable = false, flags = 131072
        : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr<[? + 4]>

    // wavefrontRowBase = bid.x * 256 + wid * 64.
    %bid256_idx     = affine.apply #map_elem(%bidx, %c0_idx)[%c256_idx]
    %wid64_idx      = affine.apply #map_elem(%wid_idx, %c0_idx)[%c64_idx]
    %wfRowBase_idx  = affine.apply #map_add(%bid256_idx, %wid64_idx)

    scf.for %r = %c0_idx to %c64_idx step %c1_idx {
      %row_idx = affine.apply #map_add(%wfRowBase_idx, %r)

      // Predicate: load is valid only when lane < n_d AND row < m.
      %lane_ok = arith.cmpi ult, %lane_idx, %n_d_idx : index
      %row_ok  = arith.cmpi ult, %row_idx,  %m_idx   : index
      %pred    = arith.andi %lane_ok, %row_ok : i1

      // Compute element offset and force OOB on invalid lanes/rows.
      // The buffer hardware bounds check drops the load silently; we
      // additionally zero the value via arith.select so the accumulation
      // is correct even for lanes that read a neighbouring live element.
      %elem_idx    = affine.apply #map_elem(%row_idx, %lane_idx)[%n_d_idx]
      %byte_off_i  = affine.apply #map_times4(%elem_idx)
      %byte_off_ok = arith.index_cast %byte_off_i : index to i32
      %byte_off    = arith.select %pred, %byte_off_ok, %cm1_i32 : i32
      %voff        = lsir.to_reg %byte_off : i32 -> !amdgcn.vgpr
      %load_dst    = amdgcn.alloca : !amdgcn.vgpr
      %val, %tok   = amdgcn.buffer_load_dword dest %load_dst addr %d_rsrc
                         offset u(%c0_s) + off_idx(%voff) + c(%c0_i32) {offen}
                         : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr)
                           mods(i32) -> !amdgcn.read_token<flat>
      %wf_ld = amdgcn.wait deps %tok
          : !amdgcn.read_token<flat> -> !amdgcn.fence_token
      %loaded_f32 = lsir.from_reg %val : !amdgcn.vgpr -> f32
      // Zero out lanes that are OOB so they don't pollute the sum.
      %v = arith.select %pred, %loaded_f32, %c0f : f32

      // Butterfly reduction: 6 constexpr rounds, strides 1, 2, 4, 8, 16, 32.
      // After 6 rounds lane 0 holds the full row sum.
      %partial_v = lsir.to_reg %v         : f32 -> !amdgcn.vgpr
      %lane_v    = lsir.to_reg %lane_i32  : i32 -> !amdgcn.vgpr
      %s6 = scf.for %round = %c0_idx to %c6_idx step %c1_idx
          iter_args(%acc = %partial_v) -> !amdgcn.vgpr {
        %stride_idx = arith.shli %c1_idx, %round : index
        %stride_i32 = arith.index_cast %stride_idx : index to i32
        %xor_v = amdgcn.alloca : !amdgcn.vgpr
        %peer  = amdgcn.v_xor_b32 outs(%xor_v) ins(%stride_i32, %lane_v)
                 : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
        %addr_v = amdgcn.alloca : !amdgcn.vgpr
        %addr   = amdgcn.v_lshlrev_b32 outs(%addr_v) ins(%c2_i32, %peer)
                  : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
        %perm_v = amdgcn.alloca : !amdgcn.vgpr
        %perm, %ptok = amdgcn.ds_bpermute_b32 outs(%perm_v) ins(%addr, %acc) args(%c0_i32)
                       : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr) args(i32)
                         -> !amdgcn.read_token<shared>
        %ns_v = amdgcn.alloca : !amdgcn.vgpr
        %ns   = amdgcn.v_add_f32 outs(%ns_v) ins(%acc, %perm)
                : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
        scf.yield %ns : !amdgcn.vgpr
      } {aster.constexpr}

      // Compute the scale rsqrt(inv_d * sum + eps) before storing so LDS holds
      // the value that phase 3 will multiply by directly.
      %sum_f32     = lsir.from_reg %s6 : !amdgcn.vgpr -> f32
      %scaled      = arith.mulf %inv_d, %sum_f32 : f32
      %shifted     = arith.addf %scaled, %eps : f32
      %shifted_v   = lsir.to_reg %shifted : f32 -> !amdgcn.vgpr
      %rsqrt_dst_v = lsir.alloca : !amdgcn.vgpr
      %rsqrt_v     = lsir.rsqrtf f32 %rsqrt_dst_v, %shifted_v : !amdgcn.vgpr, !amdgcn.vgpr

      // All lanes write the scale to LDS at (wid * 64 + r) * 4.
      // After the butterfly all lanes hold the same sum, so concurrent
      // writes to the same address are idempotent (any lane's value wins).
      %lds_elem_i = affine.apply #map_add(%wid64_idx, %r)
      %lds_byte_i = affine.apply #map_times4(%lds_elem_i)
      %lds_byte   = arith.index_cast %lds_byte_i : index to i32
      %lds_addr_v = lsir.to_reg %lds_byte : i32 -> !amdgcn.vgpr
      %wtok = amdgcn.ds_write_b32 data %rsqrt_v addr %lds_addr_v offset c(%c0_i32)
              : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
      %wf_w = amdgcn.wait deps %wtok
          : !amdgcn.write_token<shared> -> !amdgcn.fence_token
    } {aster.constexpr}

    // -------------------------------------------------------------------------
    // Phase 2: synchronise so all LDS scales are visible.
    // -------------------------------------------------------------------------
    amdgcn.s_barrier

    // -------------------------------------------------------------------------
    // Phase 3: each thread owns globalRow = bid.x * 256 + tid, reads its
    // scale from LDS[tid * 4], and multiplies its column tile of C.
    // -------------------------------------------------------------------------
    %grow_idx = affine.apply #map_add(%bid256_idx, %tid)
    %row_ok   = arith.cmpi ult, %grow_idx, %m_idx : index

    scf.if %row_ok {
      // Read this thread's pre-computed scale from LDS.
      %lds_byte_i = affine.apply #map_times4(%tid)
      %lds_byte   = arith.index_cast %lds_byte_i : index to i32
      %lds_addr_v = lsir.to_reg %lds_byte : i32 -> !amdgcn.vgpr
      %scale_dst  = amdgcn.alloca : !amdgcn.vgpr
      %scale_vgpr, %rtok = amdgcn.ds_read_b32 dest %scale_dst addr %lds_addr_v offset c(%c0_i32)
                 : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
      %wf_rs = amdgcn.wait deps %rtok
          : !amdgcn.read_token<shared> -> !amdgcn.fence_token
      %scale = lsir.from_reg %scale_vgpr : !amdgcn.vgpr -> f32

      // Iterate over this thread's 256-column tile, clamped to [0, n).
      %jStart_idx       = affine.apply #map_elem(%bidy, %c0_idx)[%c256_idx]
      %jEnd_unclamped   = affine.apply #map_add(%jStart_idx, %c256_idx)
      %jEnd_unclamped_i = arith.index_cast %jEnd_unclamped : index to i32
      %jEnd_i32         = arith.minsi %jEnd_unclamped_i, %n : i32
      %jEnd_idx         = arith.index_cast %jEnd_i32 : i32 to index

      scf.for unsigned %j = %jStart_idx to %jEnd_idx step %c1_idx {
        // C[globalRow, j] is at element offset j * m + globalRow (column-major).
        %elem_idx = affine.apply #map_elem(%j, %grow_idx)[%m_idx]
        %byte_i   = affine.apply #map_times2(%elem_idx)
        %byte_i32 = arith.index_cast %byte_i : index to i32
        %ceptr    = ptr.ptr_add %c_ptr, %byte_i32
                      : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
        %cv_bf16  = ptr.load %ceptr
                      : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> bf16
        %cv_f32   = arith.extf %cv_bf16 : bf16 to f32
        %cr_f32   = arith.mulf %cv_f32, %scale : f32
        %cr_bf16  = arith.truncf %cr_f32 : f32 to bf16
        ptr.store %cr_bf16, %ceptr
            : bf16, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      }
    }

    func.return
  }
}
