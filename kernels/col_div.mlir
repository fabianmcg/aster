// Kernel: C[i, j] = C[i, j] / sqrt(inv_d * sum(D[i, :]) + eps) for all rows i and columns j.
//
// C is column-major bf16 (2 bytes per element), i.e. C[i, j] lives at element
// offset j * m + i.  D is row-major f32 (4 bytes), i.e. D[i, k] lives at
// element offset i * n_d + k — same layout as row_div.mlir.
//
// Shape preconditions (unchecked, caller must guarantee):
//   n_d              % 16 == 0  — enables dwordx4 (4x f32) D loads.
//   rows_per_block   %  8 == 0  — every block's row range is a full
//                                 multiple of 8, so no row-band is ever
//                                 partially valid; no scalar fallback needed.
//   m % rows_per_block == 0     — combined with the above, this guarantees
//                                 every block (including the last) covers
//                                 exactly rows_per_block rows, so m itself
//                                 is also a multiple of 8. A last block
//                                 with fewer than rows_per_block rows would
//                                 produce a partial 8-row band, which the
//                                 dwordx4 OOB trick cannot safely handle
//                                 (see column-tile comment in phase 3).
//
// Algorithm:
//   Phase 1 (wave 0 only): for each row in [start_row, end_row), accumulate
//   the D row sum via a dwordx4-vectorised loop (4 f32 per load, lane stride
//   256 elements), reduce via 6-round butterfly, and write the f32 sum to
//   LDS at (row - start_row) * 4.
//   Phase 2: s_barrier so all waves see the LDS sums.
//   Phase 3 (all threads): threads are grouped 8-at-a-time (group = tid / 8,
//   lane_in_group = tid % 8). Each group cooperatively covers one 8-row band
//   of C at a time: the 8 lanes of a group hit 8 *consecutive columns* of
//   that band, each issuing one buffer_load_dwordx4 (16 bytes = 8 bf16 rows
//   of one column) — together the group's 8 loads span a contiguous
//   128-byte cache line. Groups stride across column-tiles of width 8. A
//   column-tile past n is pushed out-of-bounds via a branchless `select` on
//   the byte offset (forces the MUBUF hardware bounds check to drop the
//   access) rather than divergent control flow. bf16<->f32 conversion is
//   done by hand via shifts/masks on the raw dword bits (matching the
//   hardware sequence arith.extf/truncf itself lowers to) since a dwordx4
//   packs 2 bf16 rows per dword and there is no vector bf16<->f32 op.
//
// Shared memory: rows_per_block * 4 bytes (max 64 rows = 256 bytes).

#map_lane      = affine_map<(d0) -> (d0 mod 64)>
#map_wid       = affine_map<(d0) -> (d0 floordiv 64)>
#map_start_row = affine_map<(d0)[s0] -> (d0 * s0)>
#map_row_add   = affine_map<(d0, d1) -> (d0 + d1)>
#map_sub       = affine_map<(d0, d1) -> (d0 - d1)>
#map_elem      = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
#map_times2    = affine_map<(d0) -> (d0 * 2)>
#map_times4    = affine_map<(d0) -> (d0 * 4)>
#map_times8    = affine_map<(d0) -> (d0 * 8)>
#map_ceildiv8  = affine_map<(d0) -> (d0 ceildiv 8)>
#map_lds_byte  = affine_map<(d0, d1) -> ((d0 + d1) * 4)>

amdgcn.module @col_div_mod target = #amdgcn.target<gfx942> {
  func.func @col_div(
      %c_ptr         : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %d_ptr         : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %m             : i32,
      %n             : i32,
      %n_d           : i32,
      %rows_per_block: i32,
      %inv_d         : f32,
      %eps           : f32)
      attributes {gpu.kernel, gpu.shared_memory_size = 256 : i32} {

    %c0_i32   = arith.constant 0    : i32
    %c2_i32   = arith.constant 2    : i32
    %cm1_i32  = arith.constant -1   : i32
    %c0f      = arith.constant 0.0  : f32

    %c0_idx  = arith.constant 0  : index
    %c1_idx  = arith.constant 1  : index
    %c2_idx  = arith.constant 2  : index
    %c3_idx  = arith.constant 3  : index
    %c4_idx  = arith.constant 4  : index
    %c5_idx  = arith.constant 5  : index
    %c6_idx  = arith.constant 6  : index
    %c7_idx  = arith.constant 7  : index
    %c8_idx  = arith.constant 8  : index
    %c256_idx = arith.constant 256 : index

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
    // n_d % 16 == 0 lets every started dwordx4 load stay fully in-bounds.
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

    // Each lane loads 4 contiguous f32 (dwordx4) starting at column lane*4,
    // striding by 256 (64 lanes * 4 elems) across the row.
    %lane4_idx = affine.apply #map_times4(%lane_idx)

    scf.if %is_wave0 {
      scf.for %row = %start_row_idx to %end_row_idx step %c1_idx {
        %partial_f32 = scf.for unsigned %col = %lane4_idx to %n_d_idx step %c256_idx
            iter_args(%acc = %c0f) -> f32 {
          %elem_idx   = affine.apply #map_elem(%row, %col)[%n_d_idx]
          %byte_off_i = affine.apply #map_times4(%elem_idx)
          %byte_off   = arith.index_cast %byte_off_i : index to i32
          %voff       = lsir.to_reg %byte_off : i32 -> !amdgcn.vgpr
          %load_dst   = lsir.alloca : !amdgcn.vgpr<[? + 4]>
          %val4, %tok = amdgcn.buffer_load_dwordx4 dest %load_dst addr %d_rsrc
                            offset u(%c0_s) + off_idx(%voff) + c(%c0_i32) {offen}
                            : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr)
                              mods(i32) -> !amdgcn.read_token<flat>
          %wf_ld = amdgcn.wait deps %tok
              : !amdgcn.read_token<flat> -> !amdgcn.fence_token
          %e0, %e1, %e2, %e3 = amdgcn.split_register_range %val4 : !amdgcn.vgpr<[? + 4]>
          %f0 = lsir.from_reg %e0 : !amdgcn.vgpr -> f32
          %f1 = lsir.from_reg %e1 : !amdgcn.vgpr -> f32
          %f2 = lsir.from_reg %e2 : !amdgcn.vgpr -> f32
          %f3 = lsir.from_reg %e3 : !amdgcn.vgpr -> f32
          %a0 = arith.addf %acc, %f0 : f32
          %a1 = arith.addf %a0, %f1 : f32
          %a2 = arith.addf %a1, %f2 : f32
          %a3 = arith.addf %a2, %f3 : f32
          scf.yield %a3 : f32
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
    // Phase 3: 8-thread groups divide C via dwordx4 loads/stores.
    //
    // Buffer descriptor for the whole of C: num_records = m * n * 2 bytes,
    // stride = 0. m % 8 == 0 and rows_per_block % 8 == 0 guarantee every
    // 8-row band is fully in-bounds, so the descriptor's OOB trick only
    // needs to guard the column-tile tail (col >= n), forced via a
    // branchless select of the byte offset to -1 (0xFFFFFFFF), which is
    // always past num_records and so is dropped by the hardware bounds
    // check on both the load and the store.
    // -------------------------------------------------------------------------
    %c_total_elems   = affine.apply #map_elem(%n_idx, %c0_idx)[%m_idx]
    %c_total_bytes_i = affine.apply #map_times2(%c_total_elems)
    %c_total_bytes   = arith.index_cast %c_total_bytes_i : index to i32
    %c_sgpr          = lsir.to_reg %c_ptr : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> !amdgcn.sgpr<[? + 2]>
    %c_total_bytes_s = lsir.to_reg %c_total_bytes : i32 -> !amdgcn.sgpr
    %c_rsrc = amdgcn.make_buffer_rsrc %c_sgpr, %c_total_bytes_s, %c0_i32,
        cache_swizzle = false, swizzle_enable = false, flags = 131072
        : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr<[? + 4]>

    // group = tid / 8, lane_in_group = tid % 8 (8 is a power of two: shift/and).
    %group_idx = arith.shrui %tid, %c3_idx : index
    %lig_idx   = arith.andi  %tid, %c7_idx : index
    %lig_i32   = arith.index_cast %lig_idx : index to i32

    %n_rows_block_idx  = affine.apply #map_sub(%end_row_idx, %start_row_idx)
    %num_bands_idx     = affine.apply #map_ceildiv8(%n_rows_block_idx)
    %num_col_tiles_idx = affine.apply #map_ceildiv8(%n_idx)
    %num_groups_idx    = arith.shrui %bdim, %c3_idx : index

    // Column-tile base for this group's lane, replicated across all bands:
    // col = tile * 8 + lig. The tile loop starts at %group and strides by
    // %num_groups so groups fan out across the block's column-tile range.
    scf.for %band = %c0_idx to %num_bands_idx step %c1_idx {
      %band8_idx = affine.apply #map_times8(%band)
      %r0_idx    = affine.apply #map_row_add(%start_row_idx, %band8_idx)

      // Read the band's 8 row sums from LDS (uniform: the band's row range
      // does not depend on tid) and turn each into a reciprocal denominator
      // recip = 1 / sqrt(inv_d * sum + eps), so the per-element division
      // below becomes a multiply.
      %denom0 = func.call @col_div_denom(%band8_idx, %c0_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32
      %denom1 = func.call @col_div_denom(%band8_idx, %c1_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32
      %denom2 = func.call @col_div_denom(%band8_idx, %c2_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32
      %denom3 = func.call @col_div_denom(%band8_idx, %c3_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32
      %denom4 = func.call @col_div_denom(%band8_idx, %c4_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32
      %denom5 = func.call @col_div_denom(%band8_idx, %c5_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32
      %denom6 = func.call @col_div_denom(%band8_idx, %c6_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32
      %denom7 = func.call @col_div_denom(%band8_idx, %c7_idx, %inv_d, %eps)
          : (index, index, f32, f32) -> f32

      scf.for unsigned %tile = %group_idx to %num_col_tiles_idx step %num_groups_idx {
        %tile8_idx = affine.apply #map_times8(%tile)
        %col_idx   = affine.apply #map_row_add(%tile8_idx, %lig_idx)

        // elem = col * m + r0; byte_off = elem * 2. Force byte_off to -1
        // (past num_records) when col >= n so the buffer op is dropped.
        %elem_idx    = affine.apply #map_elem(%col_idx, %r0_idx)[%m_idx]
        %byte_off_i  = affine.apply #map_times2(%elem_idx)
        %byte_off_ok = arith.index_cast %byte_off_i : index to i32
        %col_valid   = arith.cmpi ult, %col_idx, %n_idx : index
        %byte_off    = arith.select %col_valid, %byte_off_ok, %cm1_i32 : i32
        %voff        = lsir.to_reg %byte_off : i32 -> !amdgcn.vgpr

        %load_dst   = lsir.alloca : !amdgcn.vgpr<[? + 4]>
        %val4, %ltok = amdgcn.buffer_load_dwordx4 dest %load_dst addr %c_rsrc
                          offset u(%c0_s) + off_idx(%voff) + c(%c0_i32) {offen}
                          : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr)
                            mods(i32) -> !amdgcn.read_token<flat>
        %wf_ld = amdgcn.wait deps %ltok
            : !amdgcn.read_token<flat> -> !amdgcn.fence_token
        %dw0, %dw1, %dw2, %dw3 = amdgcn.split_register_range %val4 : !amdgcn.vgpr<[? + 4]>

        // Each dword packs two consecutive rows (low 16 bits = even row,
        // high 16 bits = odd row of the band). Unpack via shifts (matching
        // the bit pattern arith.extf/truncf themselves lower to for bf16),
        // multiply by the row's reciprocal, and repack.
        %out0 = func.call @col_div_divide_dword(%dw0, %denom0, %denom1)
            : (!amdgcn.vgpr, f32, f32) -> !amdgcn.vgpr
        %out1 = func.call @col_div_divide_dword(%dw1, %denom2, %denom3)
            : (!amdgcn.vgpr, f32, f32) -> !amdgcn.vgpr
        %out2 = func.call @col_div_divide_dword(%dw2, %denom4, %denom5)
            : (!amdgcn.vgpr, f32, f32) -> !amdgcn.vgpr
        %out3 = func.call @col_div_divide_dword(%dw3, %denom6, %denom7)
            : (!amdgcn.vgpr, f32, f32) -> !amdgcn.vgpr

        %out4 = amdgcn.make_register_range %out0, %out1, %out2, %out3
            : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        %stok = amdgcn.buffer_store_dwordx4 data %out4 addr %c_rsrc
                    offset u(%c0_s) + off_idx(%voff) + c(%c0_i32) {offen}
                    : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr)
                      mods(i32) -> !amdgcn.write_token<flat>
        %wf_st = amdgcn.wait deps %stok
            : !amdgcn.write_token<flat> -> !amdgcn.fence_token
      }
    }

    func.return
  }

  // Reads the LDS row sum at (band8 + k) * 4 and returns
  // sqrt(inv_d * sum + eps).
  func.func private @col_div_denom(%band8: index, %k: index, %inv_d: f32, %eps: f32) -> f32 {
    %c0_i32 = arith.constant 0 : i32
    %r_idx  = affine.apply #map_lds_byte(%band8, %k)
    %r_byte = arith.index_cast %r_idx : index to i32
    %addr_v = lsir.to_reg %r_byte : i32 -> !amdgcn.vgpr
    %sum_dst = amdgcn.alloca : !amdgcn.vgpr
    %sum_vgpr, %rtok = amdgcn.ds_read_b32 dest %sum_dst addr %addr_v offset c(%c0_i32)
               : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
    %wf_rs = amdgcn.wait deps %rtok
        : !amdgcn.read_token<shared> -> !amdgcn.fence_token
    %sum_f32   = lsir.from_reg %sum_vgpr : !amdgcn.vgpr -> f32
    %scaled    = arith.mulf %inv_d, %sum_f32 : f32
    %shifted   = arith.addf %scaled, %eps : f32
    %shifted_v = lsir.to_reg %shifted : f32 -> !amdgcn.vgpr
    %sqrt_dst  = lsir.alloca : !amdgcn.vgpr
    %sqrt_v    = lsir.sqrtf f32 %sqrt_dst, %shifted_v : !amdgcn.vgpr, !amdgcn.vgpr
    %denom     = lsir.from_reg %sqrt_v : !amdgcn.vgpr -> f32
    func.return %denom : f32
  }

  // Divides the two bf16 rows packed in one dword by their respective
  // denominators and repacks the result into a dword. Unpacking/repacking
  // uses only shifts and masks (matching what arith.extf/truncf themselves
  // lower to for bf16<->f32) since there is no bitcast op in this pipeline
  // and a dwordx4 element carries no bf16 type information.
  func.func private @col_div_divide_dword(%dw: !amdgcn.vgpr, %denom_lo: f32, %denom_hi: f32) -> !amdgcn.vgpr {
    %c16_i32       = arith.constant 16    : i32
    %c0xffff_i32   = arith.constant 65535 : i32   // 0x0000FFFF.
    %cffff0000_i32 = arith.constant -65536 : i32  // 0xFFFF0000 as signless i32.

    %i32 = lsir.from_reg %dw : !amdgcn.vgpr -> i32

    // Low row: bf16 bits are the low 16 bits of %i32; bf16->f32 is those
    // bits shifted into the top half of a 32-bit word (sign/exponent align).
    %lo_masked = arith.andi %i32, %c0xffff_i32 : i32
    %lo_f32_i  = arith.shli %lo_masked, %c16_i32 : i32
    %lo_f32_v  = lsir.to_reg %lo_f32_i : i32 -> !amdgcn.vgpr
    %lo_f32    = lsir.from_reg %lo_f32_v : !amdgcn.vgpr -> f32
    %lo_res    = arith.divf %lo_f32, %denom_lo : f32
    %lo_res_v  = lsir.to_reg %lo_res : f32 -> !amdgcn.vgpr
    %lo_res_i  = lsir.from_reg %lo_res_v : !amdgcn.vgpr -> i32
    // f32->bf16 truncation: take the top 16 bits (no rounding).
    %lo_bf16   = arith.shrui %lo_res_i, %c16_i32 : i32

    // High row: bf16 bits are the high 16 bits of %i32; already
    // sign/exponent-aligned as an f32 once the low 16 bits are masked off.
    %hi_f32_i = arith.andi %i32, %cffff0000_i32 : i32
    %hi_f32_v = lsir.to_reg %hi_f32_i : i32 -> !amdgcn.vgpr
    %hi_f32   = lsir.from_reg %hi_f32_v : !amdgcn.vgpr -> f32
    %hi_res   = arith.divf %hi_f32, %denom_hi : f32
    %hi_res_v = lsir.to_reg %hi_res : f32 -> !amdgcn.vgpr
    %hi_res_i = lsir.from_reg %hi_res_v : !amdgcn.vgpr -> i32
    %hi_bf16  = arith.andi %hi_res_i, %cffff0000_i32 : i32

    %packed   = arith.ori %lo_bf16, %hi_bf16 : i32
    %packed_v = lsir.to_reg %packed : i32 -> !amdgcn.vgpr
    func.return %packed_v : !amdgcn.vgpr
  }
}
