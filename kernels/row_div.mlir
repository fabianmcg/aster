// Kernel: C[i, j] = C[i, j] / sqrt(sum(D[i, :])) for all rows i and columns j.
//
// C is row-major bf16 (2 bytes per element); D is row-major f32 (4 bytes).
//
// Algorithm:
//   Phase 1 (wave 0 only): for each row in [start_row, end_row), accumulate
//   the D row sum over all columns (inner loop, step 64), reduce via 6-round
//   butterfly, and write the f32 sum to LDS at (row - start_row) * 4.
//   Phase 2: s_barrier so all waves see the LDS sums.
//   Phase 3 (all threads): for each row in [start_row, end_row), read the
//   sum from LDS and divide the bf16 C elements cooperatively.
//
// No shape restrictions: n_d can exceed 64; the last block handles a partial
// set of rows via end_row = min(start_row + rows_per_block, m).
//
// Shared memory: rows_per_block * 4 bytes (max 64 rows = 256 bytes).

#map_lane      = affine_map<(d0) -> (d0 mod 64)>
#map_wid       = affine_map<(d0) -> (d0 floordiv 64)>
#map_start_row = affine_map<(d0)[s0] -> (d0 * s0)>
#map_row_add   = affine_map<(d0, d1) -> (d0 + d1)>
#map_sub       = affine_map<(d0, d1) -> (d0 - d1)>
#map_elem      = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
#map_times4    = affine_map<(d0) -> (d0 * 4)>
#map_times2    = affine_map<(d0) -> (d0 * 2)>
#map_row_byte  = affine_map<(d0)[s0] -> (d0 * s0 * 2)>

amdgcn.module @row_div_mod target = #amdgcn.target<gfx942> {
  func.func @row_div(
      %c_ptr         : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %d_ptr         : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %m             : i32,
      %n             : i32,
      %n_d           : i32,
      %rows_per_block: i32)
      attributes {gpu.kernel, gpu.shared_memory_size = 256 : i32} {

    %c0_i32 = arith.constant 0   : i32
    %c2_i32 = arith.constant 2   : i32
    %c4_i32 = arith.constant 4   : i32
    %c0f    = arith.constant 0.0 : f32

    %c0_idx  = arith.constant 0  : index
    %c1_idx  = arith.constant 1  : index
    %c6_idx  = arith.constant 6  : index
    %c64_idx = arith.constant 64 : index

    %tid  = gpu.thread_id x
    %bid  = gpu.block_id  x
    %bdim = gpu.block_dim x

    %m_idx   = arith.index_cast %m             : i32 to index
    %n_idx   = arith.index_cast %n             : i32 to index
    %n_d_idx = arith.index_cast %n_d           : i32 to index
    %rpb_idx = arith.index_cast %rows_per_block : i32 to index

    %lane_idx = affine.apply #map_lane(%tid)
    %wid_idx  = affine.apply #map_wid(%tid)
    %lane_i32 = arith.index_cast %lane_idx : index to i32

    %start_row_idx        = affine.apply #map_start_row(%bid)[%rpb_idx]
    %end_row_unclamped    = affine.apply #map_row_add(%start_row_idx, %rpb_idx)
    %end_row_unclamped_i  = arith.index_cast %end_row_unclamped : index to i32
    %end_row_i32          = arith.minsi %end_row_unclamped_i, %m : i32
    %end_row_idx          = arith.index_cast %end_row_i32 : i32 to index

    // -------------------------------------------------------------------------
    // Phase 1: wave 0 accumulates D row sums and writes them to LDS.
    // -------------------------------------------------------------------------

    // Buffer descriptor for D: num_records = m * n_d * 4 bytes, stride = 0.
    %total_elems   = affine.apply #map_elem(%m_idx, %c0_idx)[%n_d_idx]
    %total_bytes_i = affine.apply #map_times4(%total_elems)
    %total_bytes   = arith.index_cast %total_bytes_i : index to i32
    %d_sgpr        = lsir.to_reg %d_ptr       : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> !amdgcn.sgpr<[? + 2]>
    %total_bytes_s = lsir.to_reg %total_bytes  : i32 -> !amdgcn.sgpr
    %c0_s          = lsir.to_reg %c0_i32       : i32 -> !amdgcn.sgpr
    %d_rsrc = amdgcn.make_buffer_rsrc %d_sgpr, %total_bytes_s, %c0_i32,
        cache_swizzle = false, swizzle_enable = false, flags = 131072
        : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr<[? + 4]>

    // Make wave-id uniform (scalar) so scf.if doesn't create lane-divergent
    // exec-mask issues with the amdgcn register ops inside.
    %wid_i32     = arith.index_cast %wid_idx : index to i32
    %wid_v       = lsir.to_reg %wid_i32 : i32 -> !amdgcn.vgpr
    %wid_s_dst   = amdgcn.alloca : !amdgcn.sgpr
    %wid_s       = amdgcn.v_readfirstlane_b32 outs(%wid_s_dst) ins(%wid_v)
                   : outs(!amdgcn.sgpr) ins(!amdgcn.vgpr)
    %wid_scalar  = lsir.from_reg %wid_s : !amdgcn.sgpr -> i32
    %is_wave0    = arith.cmpi eq, %wid_scalar, %c0_i32 : i32

    %is_lane0 = arith.cmpi eq, %lane_i32, %c0_i32 : i32

    scf.if %is_wave0 {
      scf.for %row = %start_row_idx to %end_row_idx step %c1_idx {
        // Accumulate all columns of D[row, :] with a stride-64 loop.
        // scf.for unsigned ensures col < n_d, so no OOB masking needed.
        %partial_f32 = scf.for unsigned %col = %lane_idx to %n_d_idx step %c64_idx
            iter_args(%acc = %c0f) -> f32 {
          %elem_idx   = affine.apply #map_elem(%row, %col)[%n_d_idx]
          %byte_off_i = affine.apply #map_times4(%elem_idx)
          %byte_off   = arith.index_cast %byte_off_i : index to i32
          %voff       = lsir.to_reg %byte_off : i32 -> !amdgcn.vgpr
          %load_dst   = amdgcn.alloca : !amdgcn.vgpr
          %val, %tok  = amdgcn.buffer_load_dword dest %load_dst addr %d_rsrc
                            offset u(%c0_s) + off_idx(%voff) + c(%c0_i32) {offen}
                            : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr)
                              mods(i32) -> !amdgcn.read_token<flat>
          %wf_ld = amdgcn.wait deps %tok
              : !amdgcn.read_token<flat> -> !amdgcn.fence_token
          %val_f32 = lsir.from_reg %val : !amdgcn.vgpr -> f32
          %new_acc = arith.addf %acc, %val_f32 : f32
          scf.yield %new_acc : f32
        }

        // Butterfly reduction: 6 constexpr rounds, strides 1, 2, 4, 8, 16, 32.
        // After 6 rounds lane 0 holds the full row sum.
        %partial_v = lsir.to_reg %partial_f32 : f32 -> !amdgcn.vgpr
        %lane_v    = lsir.to_reg %lane_i32    : i32 -> !amdgcn.vgpr
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

        // Lane 0 writes the row sum to LDS at (row - start_row) * 4.
        %r_idx      = affine.apply #map_sub(%row, %start_row_idx)
        scf.if %is_lane0 {
          %lds_byte_i = affine.apply #map_times4(%r_idx)
          %lds_byte   = arith.index_cast %lds_byte_i : index to i32
          %lds_addr_v = lsir.to_reg %lds_byte : i32 -> !amdgcn.vgpr
          %wtok = amdgcn.ds_write_b32 data %s6 addr %lds_addr_v offset c(%c0_i32)
                  : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
          %wf_w = amdgcn.wait deps %wtok
              : !amdgcn.write_token<shared> -> !amdgcn.fence_token
        }
      }
    }

    // -------------------------------------------------------------------------
    // Phase 2: synchronise so LDS sums are visible to all waves.
    // -------------------------------------------------------------------------
    amdgcn.s_barrier

    // -------------------------------------------------------------------------
    // Phase 3: all threads divide their C elements by the LDS row sums.
    // -------------------------------------------------------------------------
    scf.for %row = %start_row_idx to %end_row_idx step %c1_idx {
      // Read this row's sum from LDS at (row - start_row) * 4.
      %r_idx      = affine.apply #map_sub(%row, %start_row_idx)
      %lds_byte_i = affine.apply #map_times4(%r_idx)
      %lds_byte   = arith.index_cast %lds_byte_i : index to i32
      %lds_addr_v = lsir.to_reg %lds_byte : i32 -> !amdgcn.vgpr
      %sum_dst = amdgcn.alloca : !amdgcn.vgpr
      %sum_vgpr, %rtok = amdgcn.ds_read_b32 dest %sum_dst addr %lds_addr_v offset c(%c0_i32)
                 : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
      %wf_rs = amdgcn.wait deps %rtok
          : !amdgcn.read_token<shared> -> !amdgcn.fence_token
      %sqrt_dst_v = lsir.alloca : !amdgcn.vgpr
      %sqrt_v     = lsir.sqrtf f32 %sqrt_dst_v, %sum_vgpr : !amdgcn.vgpr, !amdgcn.vgpr
      %total_f32  = lsir.from_reg %sqrt_v : !amdgcn.vgpr -> f32

      // Pointer to the start of C[row, :].
      %c_row_byte_i = affine.apply #map_row_byte(%row)[%n_idx]
      %c_row_byte   = arith.index_cast %c_row_byte_i : index to i32
      %c_row_ptr    = ptr.ptr_add %c_ptr, %c_row_byte
                        : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32

      // Each thread strides across columns with step bdim.
      scf.for unsigned %j = %tid to %n_idx step %bdim {
        %c_off_i = affine.apply #map_times2(%j)
        %c_off   = arith.index_cast %c_off_i : index to i32
        %ceptr   = ptr.ptr_add %c_row_ptr, %c_off
                     : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
        %cv_bf16 = ptr.load %ceptr
                     : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> bf16
        %cv_f32  = arith.extf %cv_bf16 : bf16 to f32
        %cr_f32  = arith.divf %cv_f32, %total_f32 : f32
        %cr_bf16 = arith.truncf %cr_f32 : f32 to bf16
        ptr.store %cr_bf16, %ceptr
            : bf16, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      }
    }

    func.return
  }
}
