// Kernel: C[i, j] = C[i, j] / sum(D[i, :]) for all rows i and columns j.
//
// Launch: grid_dim=(M,1,1), block_dim=(64,1,1) — one wavefront per row.
//
// Algorithm:
//   1. Each lane loads D[row, tid], D[row, tid+64], ... and accumulates (f32).
//   2. Butterfly warp reduction via ds_bpermute_b32 (6 rounds) → lane 0 holds sum.
//   3. Broadcast sum to all lanes via ds_bpermute_b32 with addr=0.
//   4. Each lane loads C[row, j] (bf16), extends to f32, divides, truncates
//      back to bf16, and stores.
//
// C is row-major bf16 (2 bytes per element); D is row-major f32 (4 bytes).

amdgcn.module @row_div_mod target = #amdgcn.target<gfx942> {

  func.func @row_div(
      %c_ptr : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %d_ptr : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %n     : i32,
      %n_d   : i32)
      attributes {gpu.kernel} {

    %tid  = aster_utils.thread_id x
    %row  = aster_utils.block_id x

    %c0f  = arith.constant 0.0 : f32
    %c4   = arith.constant 4   : i32
    %c64  = arith.constant 64  : i32
    %c2   = arith.constant 2   : i32
    %c0   = arith.constant 0   : i32

    // -------------------------------------------------------------------------
    // Phase 1: accumulate partial row sum of D.
    // Each lane sums D[row, tid], D[row, tid+64], ...
    // -------------------------------------------------------------------------

    %d_row_elems = arith.muli %row, %n_d : i32
    %d_row_byte  = arith.muli %d_row_elems, %c4 : i32
    %d_row_ptr   = ptr.ptr_add %d_ptr, %d_row_byte
                     : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32

    %partial = scf.for unsigned %k = %tid to %n_d step %c64
        iter_args(%acc = %c0f) -> f32 : i32 {
      %boff = arith.muli %k, %c4 : i32
      %dptr = ptr.ptr_add %d_row_ptr, %boff
                : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
      %val  = ptr.load %dptr
                : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> f32
      %nacc = arith.addf %acc, %val : f32
      scf.yield %nacc : f32
    }

    // -------------------------------------------------------------------------
    // Phase 2: butterfly warp reduction via ds_bpermute_b32.
    // ds_bpermute_b32: VDST[i] = DATA[(ADDR[i]/4) % 64]  (gather).
    //
    // The loop runs 6 rounds (log2(64)) with aster.constexpr, so the compiler
    // fully unrolls it at compile time. At each unrolled iteration %round is a
    // constant, making stride = 1 << round fold to 1, 2, 4, 8, 16, 32.
    // After 6 rounds, lane 0 holds the total sum of all 64 lanes.
    // -------------------------------------------------------------------------

    // Lift f32 partial sum into a VGPR and thread ID into a VGPR.
    %acc_init = lsir.to_reg %partial : f32 -> !amdgcn.vgpr
    %tid_v    = lsir.to_reg %tid     : i32 -> !amdgcn.vgpr

    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c6_idx = arith.constant 6 : index

    // iter_arg carries the running VGPR sum across rounds.
    %s5 = scf.for %round = %c0_idx to %c6_idx step %c1_idx
        iter_args(%sum = %acc_init) -> !amdgcn.vgpr {
      // stride = 1 << round  (index → i32 constant after unrolling).
      %stride_idx = arith.shli %c1_idx, %round : index
      %stride_i32 = arith.index_cast %stride_idx : index to i32
      // peer lane = tid XOR stride; ds_bpermute addr = peer * 4.
      %xor_v  = amdgcn.alloca : !amdgcn.vgpr
      %peer   = amdgcn.v_xor_b32 outs(%xor_v) ins(%stride_i32, %tid_v)
                  : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
      %addr_v = amdgcn.alloca : !amdgcn.vgpr
      %addr   = amdgcn.v_lshlrev_b32 outs(%addr_v) ins(%c2, %peer)
                  : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
      // Gather from the peer lane, wait, add.
      %perm_v = amdgcn.alloca : !amdgcn.vgpr
      %perm, %tok = amdgcn.ds_bpermute_b32 outs(%perm_v) ins(%addr, %sum) args(%c0)
                      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr) args(i32)
                        -> !amdgcn.read_token<shared>
      %new_sum_v = amdgcn.alloca : !amdgcn.vgpr
      %new_sum   = amdgcn.v_add_f32 outs(%new_sum_v) ins(%sum, %perm)
                     : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
      scf.yield %new_sum : !amdgcn.vgpr
    } {aster.constexpr}

    // -------------------------------------------------------------------------
    // Phase 3: broadcast total sum from lane 0 to all lanes.
    // All lanes use addr=0, so ds_bpermute reads from lane 0.
    // -------------------------------------------------------------------------
    %bc_v    = amdgcn.alloca : !amdgcn.vgpr
    %bc_addr = amdgcn.v_mov_b32 outs(%bc_v) ins(%c0)
                 : outs(!amdgcn.vgpr) ins(i32)
    %tot_v   = amdgcn.alloca : !amdgcn.vgpr
    %tot, %t6 = amdgcn.ds_bpermute_b32 outs(%tot_v) ins(%bc_addr, %s5) args(%c0)
                  : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr) args(i32)
                    -> !amdgcn.read_token<shared>

    // Convert the total sum VGPR back to f32 for the division loop.
    %total_f32 = lsir.from_reg %tot : !amdgcn.vgpr -> f32

    // -------------------------------------------------------------------------
    // Phase 4: divide each C[row, j] (bf16) by the total row sum.
    // Each lane handles j = tid, tid+64, tid+128, ...
    // C uses 2-byte elements; byte offset = j * 2.
    // -------------------------------------------------------------------------
    %c_row_elems = arith.muli %row, %n : i32
    %c_row_byte  = arith.muli %c_row_elems, %c2 : i32
    %c_row_ptr   = ptr.ptr_add %c_ptr, %c_row_byte
                     : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32

    scf.for unsigned %j = %tid to %n step %c64 : i32 {
      %boff  = arith.muli %j, %c2 : i32
      %ceptr = ptr.ptr_add %c_row_ptr, %boff
                 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
      // Load bf16, extend to f32, divide, truncate back to bf16, store.
      %cval_bf16 = ptr.load %ceptr
                     : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> bf16
      %cval_f32  = arith.extf %cval_bf16 : bf16 to f32
      %cnew_f32  = arith.divf %cval_f32, %total_f32 : f32
      %cnew_bf16 = arith.truncf %cnew_f32 : f32 to bf16
      ptr.store %cnew_bf16, %ceptr
                  : bf16, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }

    func.return
  }
}
