// RUN: aster-opt %s --verify-roundtrip --split-input-file
//
// E2E tests for scf.while lowering via aster-convert-scf-control-flow.
//
// Three kernels, each in its own module:
//   test_while_iter_args  -- accumulate sum(0..7) with iter_args; store 28.
//   test_while_divergent  -- lane i runs i iterations; store tid.
//   test_while_zero_trip  -- condition false on entry; init value stored.

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 1 x i32 pre-filled with 0.
// Expected: output[0] = 28  (sum of 0+1+2+3+4+5+6+7).
// All lanes write 28 to the same slot (uniform scalar result).
amdgcn.module @scf_while_iter_args_mod target = <gfx942> {
  func.func @test_while_iter_args(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    // scf.while with two iter_args: (%i, %acc).
    // Condition: %i < 8. Body: acc += i; i += 1.
    // After the loop acc = 0+1+...+7 = 28.
    %_i, %sum = scf.while (%i = %c0, %acc = %c0) : (i32, i32) -> (i32, i32) {
      %cond = arith.cmpi slt, %i, %c8 : i32
      scf.condition(%cond) %i, %acc : i32, i32
    } do {
    ^bb0(%cur_i: i32, %cur_acc: i32):
      %new_acc = arith.addi %cur_acc, %cur_i : i32
      %new_i   = arith.addi %cur_i, %c1 : i32
      scf.yield %new_i, %new_acc : i32, i32
    }
    // All lanes write the same sum to output[0].
    ptr.store %sum, %buf : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[i] = i.
//
// Lane i runs the loop body i times (condition: cnt < tid).
// After the loop, EXEC is restored to all 64 lanes and each lane stores its tid.
amdgcn.module @scf_while_divergent_mod target = <gfx942> {
  func.func @test_while_divergent(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid = aster_utils.thread_id x
    %c0  = arith.constant 0 : i32
    %c1  = arith.constant 1 : i32
    %c4  = arith.constant 4 : i32
    // Divergent loop: lane i runs i iterations.
    %_cnt = scf.while (%cnt = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %cnt, %tid : i32
      scf.condition(%cond) %cnt : i32
    } do {
    ^bb0(%cur: i32):
      %next = arith.addi %cur, %c1 : i32
      scf.yield %next : i32
    }
    // After the loop, all lanes are active (EXEC restored). Store tid per lane.
    %off = arith.muli %tid, %c4 : i32
    %ptr = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    ptr.store %tid, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 1 x i32 pre-filled with 99.
// Expected: output[0] = 99 (init value unchanged; condition false on first check).
//
// Zero-trip scf.while: before-block condition evaluates false immediately,
// forwarding the init value directly to the exit without executing the body.
amdgcn.module @scf_while_zero_trip_mod target = <gfx942> {
  func.func @test_while_zero_trip(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %c0  = arith.constant 0 : i32
    %c1  = arith.constant 1 : i32
    %c99 = arith.constant 99 : i32
    // scf.while: condition is cnt < 0, which is false from the start.
    // The init value 99 flows through conditionArgs to the exit unchanged.
    %result = scf.while (%cnt = %c99) : (i32) -> i32 {
      %cond = arith.cmpi slt, %cnt, %c0 : i32
      scf.condition(%cond) %cnt : i32
    } do {
    ^bb0(%cur: i32):
      // Never executed.
      %next = arith.addi %cur, %c1 : i32
      scf.yield %next : i32
    }
    // Store the init value: condition is always false so %result = %c99.
    ptr.store %result, %buf : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    return
  }
}
