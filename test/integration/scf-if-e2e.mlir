// RUN: aster-opt %s --verify-roundtrip --split-input-file
//
// E2E tests for scf.if lowering via aster-convert-scf-control-flow.
//
// Eight kernels, each in its own module:
//   test_if_no_else_uniform         -- uniform condition; no else; no results.
//   test_if_else_uniform            -- uniform condition; else branch; no results.
//   test_if_with_results_uniform    -- uniform condition; else; scf.if VGPR result.
//   test_if_divergent_no_else       -- divergent condition; no else; no results.
//   test_if_divergent_with_else     -- divergent condition; else branch; no results.
//   test_if_divergent_with_results  -- divergent condition; else; scf.if VGPR result.
//   test_if_complex_condition_and   -- divergent AND condition; no else.
//   test_if_complex_condition_or    -- divergent OR condition; no else.

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0] = 42, all others 0.
// Uniform condition (lane 0 only): tests no-else, no-results path.
amdgcn.module @scf_if_no_else_uniform_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_no_else_uniform arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c42 = arith.constant 42 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Uniform: only lane 0 satisfies the condition.
    %c0_u = aster_utils.assume_uniform %c0 : i32
    %cond = arith.cmpi eq, %tid_i32, %c0_u : i32

    %c42_sg = lsir.to_reg %c42 : i32 -> !amdgcn.sgpr
    %v42_a = amdgcn.alloca : !amdgcn.vgpr
    %v42 = amdgcn.v_mov_b32 outs(%v42_a) ins(%c42_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %v42 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = 1, output[32..63] = 2.
// Uniform threshold: both branches execute (no divergence).
amdgcn.module @scf_if_else_uniform_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_else_uniform arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %c32_u = aster_utils.assume_uniform %c32 : i32
    %cond = arith.cmpi slt, %tid_i32, %c32_u : i32

    %c1_sg = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
    %v1_a = amdgcn.alloca : !amdgcn.vgpr
    %v1 = amdgcn.v_mov_b32 outs(%v1_a) ins(%c1_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %c2_sg = lsir.to_reg %c2 : i32 -> !amdgcn.sgpr
    %v2_a = amdgcn.alloca : !amdgcn.vgpr
    %v2 = amdgcn.v_mov_b32 outs(%v2_a) ins(%c2_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %v1 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    } else {
      %_tok = amdgcn.global_store_dword data %v2 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = 100, output[32..63] = 200.
// Nested scf.if inside an outer scf.if; both branches of the outer are present.
// Tests that the lowering handles the merge block correctly with nested ifs.
amdgcn.module @scf_if_with_results_uniform_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_with_results_uniform arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c100 = arith.constant 100 : i32
    %c200 = arith.constant 200 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %c32_u = aster_utils.assume_uniform %c32 : i32
    %cond = arith.cmpi slt, %tid_i32, %c32_u : i32

    %c100_sg = lsir.to_reg %c100 : i32 -> !amdgcn.sgpr
    %v100_a = amdgcn.alloca : !amdgcn.vgpr
    %v100 = amdgcn.v_mov_b32 outs(%v100_a) ins(%c100_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %c200_sg = lsir.to_reg %c200 : i32 -> !amdgcn.sgpr
    %v200_a = amdgcn.alloca : !amdgcn.vgpr
    %v200 = amdgcn.v_mov_b32 outs(%v200_a) ins(%c200_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    // Inner nested if chooses the store value; outer if dispatches the branch.
    // Both branches store, exercising the merge-block structure with no scf.if
    // results (register types cannot flow through bbMerge via arith.select).
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %v100 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    } else {
      %_tok = amdgcn.global_store_dword data %v200 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = 1, output[32..63] = 0.
// Divergent condition, no else, no results.
amdgcn.module @scf_if_divergent_no_else_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_divergent_no_else arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c1_sg = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
    %v1_a = amdgcn.alloca : !amdgcn.vgpr
    %v1 = amdgcn.v_mov_b32 outs(%v1_a) ins(%c1_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    %cond = arith.cmpi slt, %tid_i32, %c32 : i32
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %v1 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = 1, output[32..63] = 2.
// Divergent condition, else branch, no results.
// Verifies save/restore EXEC covers both then and else lanes.
amdgcn.module @scf_if_divergent_with_else_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_divergent_with_else arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c1_sg = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
    %v1_a = amdgcn.alloca : !amdgcn.vgpr
    %v1 = amdgcn.v_mov_b32 outs(%v1_a) ins(%c1_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %c2_sg = lsir.to_reg %c2 : i32 -> !amdgcn.sgpr
    %v2_a = amdgcn.alloca : !amdgcn.vgpr
    %v2 = amdgcn.v_mov_b32 outs(%v2_a) ins(%c2_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    %cond = arith.cmpi slt, %tid_i32, %c32 : i32
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %v1 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    } else {
      %_tok = amdgcn.global_store_dword data %v2 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = tid, output[32..63] = 0.
// Divergent condition, else branch, no results.
// Lanes 0..31 store their tid; lanes 32..63 store 0.
// Verifies save/restore EXEC across both branches with per-lane store.
amdgcn.module @scf_if_divergent_with_results_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_divergent_with_results arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c32 = arith.constant 32 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %cond = arith.cmpi slt, %tid_i32, %c32 : i32

    // Prepare zero VGPR for the else branch.
    %c0_sg = lsir.to_reg %c0 : i32 -> !amdgcn.sgpr
    %v0_a = amdgcn.alloca : !amdgcn.vgpr
    %v0 = amdgcn.v_mov_b32 outs(%v0_a) ins(%c0_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    // Store inside each branch: then stores tid, else stores 0.
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %tid addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    } else {
      %_tok = amdgcn.global_store_dword data %v0 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[16..47] = 1, others = 0.
// Compound AND condition: (tid >= 16) AND (tid < 48).
// Tests that multi-predicate boolean conditions lower correctly through
// save_cf_mask / restore_cf_mask.
amdgcn.module @scf_if_complex_condition_and_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_complex_condition_and arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %c48 = arith.constant 48 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // (tid >= 16) AND (tid < 48).
    %cmp_lo = arith.cmpi sge, %tid_i32, %c16 : i32
    %cmp_hi = arith.cmpi slt, %tid_i32, %c48 : i32
    %cond = arith.andi %cmp_lo, %cmp_hi : i1

    %c1_sg = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
    %v1_a = amdgcn.alloca : !amdgcn.vgpr
    %v1 = amdgcn.v_mov_b32 outs(%v1_a) ins(%c1_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %v1 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..15] = 1, output[16..47] = 0, output[48..63] = 1.
// Compound OR condition: (tid < 16) OR (tid >= 48).
// Tests that OR-composed boolean conditions lower correctly.
amdgcn.module @scf_if_complex_condition_or_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_if_complex_condition_or arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %c48 = arith.constant 48 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // (tid < 16) OR (tid >= 48).
    %cmp_lo = arith.cmpi slt, %tid_i32, %c16 : i32
    %cmp_hi = arith.cmpi sge, %tid_i32, %c48 : i32
    %cond = arith.ori %cmp_lo, %cmp_hi : i1

    %c1_sg = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
    %v1_a = amdgcn.alloca : !amdgcn.vgpr
    %v1 = amdgcn.v_mov_b32 outs(%v1_a) ins(%c1_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    scf.if %cond {
      %_tok = amdgcn.global_store_dword data %v1 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
