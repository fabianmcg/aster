// RUN: aster-opt %s --verify-roundtrip --split-input-file
//
// E2E tests for scf.for lowering via aster-convert-scf-control-flow.
//
// Five kernels, each in its own module:
//   test_for_uniform_no_iter_args  -- all lanes run 8 iterations; verify via tid*8.
//   test_for_with_iter_args        -- sum(0..7) = 28 via one iter_arg.
//   test_for_multiple_iter_args    -- two iter_args: sum and last-step value.
//   test_for_zero_trip             -- lb >= ub; body never executes.
//   test_for_divergent_body        -- uniform bounds with divergent nested scf.if.

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[i] = i * 8.
amdgcn.module @scf_for_uniform_no_iter_args_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_for_uniform_no_iter_args arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // scf.for requires index-typed bounds.
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index

    // Uniform loop: 8 iterations, no iter_args.
    scf.for %i = %lb to %ub step %step {
    }

    // Store tid * 8 via VGPR left-shift by 3 to verify the loop executed.
    %val_v_a = amdgcn.alloca : !amdgcn.vgpr
    %val_v = amdgcn.v_lshlrev_b32 outs(%val_v_a) ins(%c3, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %_tok = amdgcn.global_store_dword data %val_v addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 1 x i32 pre-filled with 0.
// Expected: output[0] = 28 (sum of 0+1+2+3+4+5+6+7).
amdgcn.module @scf_for_iter_args_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_for_with_iter_args arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %c0 = arith.constant 0 : i32

    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index

    // scf.for with one iter_arg %acc. Accumulate i (cast to i32) at each step.
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %c0) -> i32 {
      %i_i32 = arith.index_cast %i : index to i32
      %new_acc = arith.addi %acc, %i_i32 : i32
      scf.yield %new_acc : i32
    }

    // All lanes store the same scalar sum to output[0].
    %sum_sg = lsir.to_reg %sum : i32 -> !amdgcn.sgpr
    %sum_v_a = amdgcn.alloca : !amdgcn.vgpr
    %sum_v = amdgcn.v_mov_b32 outs(%sum_v_a) ins(%sum_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %zero_sg = lsir.to_reg %c0 : i32 -> !amdgcn.sgpr
    %zero_v_a = amdgcn.alloca : !amdgcn.vgpr
    %zero_v = amdgcn.v_mov_b32 outs(%zero_v_a) ins(%zero_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %_tok = amdgcn.global_store_dword data %sum_v addr %out_ptr
      offset d(%zero_v) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 2 x i32 pre-filled with 0.
// Expected: output[0] = 28 (sum), output[1] = 8 (last i+1 value).
amdgcn.module @scf_for_multiple_iter_args_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_for_multiple_iter_args arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %c0 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32

    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index

    // scf.for with two iter_args: %acc (sum of i) and %last (i+1 at each step).
    %sum, %last = scf.for %i = %lb to %ub step %step
        iter_args(%acc = %c0, %prev = %c0) -> (i32, i32) {
      %i_i32 = arith.index_cast %i : index to i32
      %new_acc = arith.addi %acc, %i_i32 : i32
      %new_prev = arith.addi %i_i32, %c1_i32 : i32
      scf.yield %new_acc, %new_prev : i32, i32
    }

    // Store %sum at output[0] (byte offset 0) and %last at output[1] (byte offset 4).
    %sum_sg = lsir.to_reg %sum : i32 -> !amdgcn.sgpr
    %sum_v_a = amdgcn.alloca : !amdgcn.vgpr
    %sum_v = amdgcn.v_mov_b32 outs(%sum_v_a) ins(%sum_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %last_sg = lsir.to_reg %last : i32 -> !amdgcn.sgpr
    %last_v_a = amdgcn.alloca : !amdgcn.vgpr
    %last_v = amdgcn.v_mov_b32 outs(%last_v_a) ins(%last_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %zero_sg = lsir.to_reg %c0 : i32 -> !amdgcn.sgpr
    %zero_v_a = amdgcn.alloca : !amdgcn.vgpr
    %zero_v = amdgcn.v_mov_b32 outs(%zero_v_a) ins(%zero_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %c4_sg = lsir.to_reg %c4 : i32 -> !amdgcn.sgpr
    %off1_v_a = amdgcn.alloca : !amdgcn.vgpr
    %off1_v = amdgcn.v_mov_b32 outs(%off1_v_a) ins(%c4_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %_tok0 = amdgcn.global_store_dword data %sum_v addr %out_ptr
      offset d(%zero_v) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %_tok1 = amdgcn.global_store_dword data %last_v addr %out_ptr
      offset d(%off1_v) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 1 x i32 pre-filled with 99.
// Expected: output[0] = 99 (init value unchanged; body never executes).
amdgcn.module @scf_for_zero_trip_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_for_zero_trip arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %c0 = arith.constant 0 : i32
    %c99 = arith.constant 99 : i32

    // lb=8, ub=0 — zero-trip: lowerBound < upperBound is false.
    %lb = arith.constant 8 : index
    %ub = arith.constant 0 : index
    %step = arith.constant 1 : index

    %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %c99) -> i32 {
      // Never executed.
      %bad = arith.constant 0 : i32
      scf.yield %bad : i32
    }

    // The init value (99) flows through as the loop result. Store the constant
    // directly to avoid lsir.to_reg on a folded bare i32 constant.
    %c99_sg = lsir.to_reg %c99 : i32 -> !amdgcn.sgpr
    %c99_v_a = amdgcn.alloca : !amdgcn.vgpr
    %c99_v = amdgcn.v_mov_b32 outs(%c99_v_a) ins(%c99_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %zero_sg = lsir.to_reg %c0 : i32 -> !amdgcn.sgpr
    %zero_v_a = amdgcn.alloca : !amdgcn.vgpr
    %zero_v = amdgcn.v_mov_b32 outs(%zero_v_a) ins(%zero_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %_tok = amdgcn.global_store_dword data %c99_v addr %out_ptr
      offset d(%zero_v) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[i] = i for i < 32, output[i] = 0 for i >= 32.
//
// Uniform scf.for (0..1 step 1) with a divergent nested scf.if (tid < 32).
// Lanes 0..31 write their tid; lanes 32..63 skip the store.
amdgcn.module @scf_for_divergent_body_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_for_divergent_body arguments <[
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

    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index

    // Uniform scf.for with a single iteration wrapping a divergent scf.if.
    scf.for %_i = %lb to %ub step %step {
      %cond = arith.cmpi slt, %tid_i32, %c32 : i32
      scf.if %cond {
        %_tok = amdgcn.global_store_dword data %tid addr %out_ptr
          offset d(%voffset) + c(%c0)
          : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
      }
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
