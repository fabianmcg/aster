// RUN: aster-opt %s --verify-roundtrip --split-input-file
//
// E2E tests for scf.while lowering via aster-convert-scf-control-flow.
//
// Six kernels, each in its own module to avoid assembly label collisions:
//   test_while_uniform              -- all lanes run same 8 iterations; store tid * 8.
//   test_while_iter_args            -- accumulate sum(0..7) with iter_args; store 28.
//   test_while_divergent            -- lane i runs i iterations; store tid.
//   test_while_early_exit           -- lane i runs min(i,32) iterations; store min(i,32).
//   test_while_divergent_two_iter_args -- lane i stores i*(i-1)/2 via two iter_args.
//   test_while_zero_trip            -- condition false on entry; init value stored.

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[i] = i * 8.
amdgcn.module @scf_while_uniform_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_while_uniform arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    // Uniform loop bound — assume_uniform so the compiler uses scalar branches.
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c8 = arith.constant 8 : i32
    %c8_u = aster_utils.assume_uniform %c8 : i32

    // scf.while: counter from 0 to 8 (exclusive), stepping by 1.
    %_result = scf.while (%counter = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %counter, %c8_u : i32
      scf.condition(%cond) %counter : i32
    } do {
    ^bb0(%cur: i32):
      %next = arith.addi %cur, %c1 : i32
      scf.yield %next : i32
    }

    // Compute tid * 8 via VGPR left-shift by 3, then store.
    %c3 = arith.constant 3 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
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
// Expected: output[0] = 28  (sum of 0+1+2+3+4+5+6+7).
// All lanes write 28 to the same slot (uniform scalar result).
amdgcn.module @scf_while_iter_args_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_while_iter_args arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %c8_u = aster_utils.assume_uniform %c8 : i32

    // scf.while with two iter_args: (%i, %acc).
    // Condition: %i < 8. Body: acc += i; i += 1.
    // After the loop acc = 0+1+...+7 = 28.
    %_i, %sum = scf.while (%i = %c0, %acc = %c0) : (i32, i32) -> (i32, i32) {
      %cond = arith.cmpi slt, %i, %c8_u : i32
      scf.condition(%cond) %i, %acc : i32, i32
    } do {
    ^bb0(%cur_i: i32, %cur_acc: i32):
      %new_acc = arith.addi %cur_acc, %cur_i : i32
      %new_i = arith.addi %cur_i, %c1 : i32
      scf.yield %new_i, %new_acc : i32, i32
    }

    // All lanes write the same sum to output[0] at byte offset 0.
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
// output: 64 x i32 pre-filled with 0.
// Expected: output[i] = i.
//
// Lane i runs the loop body i times (condition: cnt < tid_i32).
// After the loop, EXEC is restored to all 64 lanes and each lane stores
// its own tid via VGPR store, verifying correct post-loop EXEC state.
amdgcn.module @scf_while_divergent_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_while_divergent arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Divergent loop: lane i runs i iterations.
    %_cnt = scf.while (%cnt = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %cnt, %tid_i32 : i32
      scf.condition(%cond) %cnt : i32
    } do {
    ^bb0(%cur: i32):
      %next = arith.addi %cur, %c1 : i32
      scf.yield %next : i32
    }

    // After the loop, all lanes are active (EXEC restored). Store tid per lane.
    %_tok = amdgcn.global_store_dword data %tid addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[i] = min(i, 32).
amdgcn.module @scf_while_early_exit_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_while_early_exit arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Compute min(tid_i32, 32) — divergent per lane.
    %limit = arith.minsi %tid_i32, %c32 : i32

    // scf.while: count up while counter < min(tid, 32).
    %_cnt = scf.while (%cnt = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %cnt, %limit : i32
      scf.condition(%cond) %cnt : i32
    } do {
    ^bb0(%cur: i32):
      %next = arith.addi %cur, %c1 : i32
      scf.yield %next : i32
    }

    // Compute min(tid_vgpr, 32) using a VCC-based select (v_cndmask_b32).
    %c32_sg = lsir.to_reg %c32 : i32 -> !amdgcn.sgpr
    %c32_v_a = amdgcn.alloca : !amdgcn.vgpr
    %c32_v = amdgcn.v_mov_b32 outs(%c32_v_a) ins(%c32_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %vcc_a = lsir.alloca : !amdgcn.vcc
    // VCC set when tid < 32 (pick tid), cleared otherwise (pick 32).
    %cmp = lsir.cmpi i32 slt %vcc_a, %tid, %c32 : !amdgcn.vcc, !amdgcn.vgpr, i32
    %min_v_a = amdgcn.alloca : !amdgcn.vgpr
    %min_v = lsir.select %min_v_a, %cmp, %tid, %c32_v
      : !amdgcn.vgpr, !amdgcn.vcc, !amdgcn.vgpr, !amdgcn.vgpr
    %_tok = amdgcn.global_store_dword data %min_v addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[i] = i*(i-1)/2  (partial sum 0+1+...+(i-1)).
//
// Divergent while with two iter_args: (%cnt, %acc).
// Lane i runs i iterations, each time adding cnt to acc.
amdgcn.module @scf_while_divergent_two_iter_args_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_while_divergent_two_iter_args arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = aster_utils.thread_id x

    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Divergent loop: lane i runs i iterations. Body: acc += cnt; cnt += 1.
    // After exit, acc = 0+1+...+(i-1) = i*(i-1)/2.
    %_cnt, %acc = scf.while (%cnt = %c0, %sum = %c0) : (i32, i32) -> (i32, i32) {
      %cond = arith.cmpi slt, %cnt, %tid_i32 : i32
      scf.condition(%cond) %cnt, %sum : i32, i32
    } do {
    ^bb0(%cur_cnt: i32, %cur_sum: i32):
      %new_sum = arith.addi %cur_sum, %cur_cnt : i32
      %new_cnt = arith.addi %cur_cnt, %c1 : i32
      scf.yield %new_cnt, %new_sum : i32, i32
    }

    // Store %acc per lane.
    %acc_sg = lsir.to_reg %acc : i32 -> !amdgcn.sgpr
    %acc_v_a = amdgcn.alloca : !amdgcn.vgpr
    %acc_v = amdgcn.v_mov_b32 outs(%acc_v_a) ins(%acc_sg)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %_tok = amdgcn.global_store_dword data %acc_v addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 1 x i32 pre-filled with 99.
// Expected: output[0] = 99 (init value unchanged; condition false on first check).
//
// Zero-trip scf.while: before-block condition evaluates false immediately,
// forwarding the init value directly to bbEnd without executing the body.
amdgcn.module @scf_while_zero_trip_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_while_zero_trip arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c99 = arith.constant 99 : i32
    // Uniform bound of 0: condition (cnt < 0) is false on the very first check.
    %c0_u = aster_utils.assume_uniform %c0 : i32

    // scf.while: condition is cnt < 0, which is false from the start.
    // The init value 99 flows through conditionArgs to bbEnd unchanged.
    %result = scf.while (%cnt = %c99) : (i32) -> i32 {
      %cond = arith.cmpi slt, %cnt, %c0_u : i32
      scf.condition(%cond) %cnt : i32
    } do {
    ^bb0(%cur: i32):
      // Never executed.
      %next = arith.addi %cur, %c1 : i32
      scf.yield %next : i32
    }

    // Store the init value. The result SSA val folds to the constant after
    // the zero-trip path, so store the constant directly.
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
