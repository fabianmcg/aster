// RUN: aster-opt %s --verify-roundtrip --split-input-file
//
// E2E tests for scf.for lowering via aster-convert-scf-control-flow.
//
// Three kernels, each in its own module:
//   test_for_with_iter_args        -- sum(0..7) = 28 via one iter_arg.
//   test_for_multiple_iter_args    -- two iter_args: sum and last-step value.
//   test_for_zero_trip             -- lb >= ub; body never executes.

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 1 x i32 pre-filled with 0.
// Expected: output[0] = 28 (sum of 0+1+2+3+4+5+6+7).
amdgcn.module @scf_for_iter_args_mod target = <gfx942> {
  func.func @test_for_with_iter_args(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %c0  = arith.constant 0 : i32
    %c0i = arith.constant 0 : index
    %c8i = arith.constant 8 : index
    %c1i = arith.constant 1 : index
    // scf.for with one iter_arg %acc. Accumulate i (cast to i32) at each step.
    %sum = scf.for %i = %c0i to %c8i step %c1i iter_args(%acc = %c0) -> i32 {
      %i_i32   = arith.index_cast %i : index to i32
      %new_acc = arith.addi %acc, %i_i32 : i32
      scf.yield %new_acc : i32
    }
    // All lanes store the same scalar sum to output[0].
    ptr.store %sum, %buf : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 2 x i32 pre-filled with 0.
// Expected: output[0] = 28 (sum), output[1] = 8 (last i+1 value).
amdgcn.module @scf_for_multiple_iter_args_mod target = <gfx942> {
  func.func @test_for_multiple_iter_args(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %c0  = arith.constant 0 : i32
    %c1  = arith.constant 1 : i32
    %c0i = arith.constant 0 : index
    %c8i = arith.constant 8 : index
    %c1i = arith.constant 1 : index
    // scf.for with two iter_args: %acc (sum of i) and %last (i+1 at each step).
    %sum, %last = scf.for %i = %c0i to %c8i step %c1i
        iter_args(%acc = %c0, %prev = %c0) -> (i32, i32) {
      %i_i32    = arith.index_cast %i : index to i32
      %new_acc  = arith.addi %acc, %i_i32 : i32
      %new_prev = arith.addi %i_i32, %c1 : i32
      scf.yield %new_acc, %new_prev : i32, i32
    }
    // Store %sum at output[0] and %last at output[1].
    %c4   = arith.constant 4 : i32
    %ptr1 = ptr.ptr_add %buf, %c4 : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    ptr.store %sum, %buf : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    ptr.store %last, %ptr1 : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 1 x i32 pre-filled with 99.
// Expected: output[0] = 99 (init value unchanged; body never executes).
amdgcn.module @scf_for_zero_trip_mod target = <gfx942> {
  func.func @test_for_zero_trip(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %c99 = arith.constant 99 : i32
    // lb=8, ub=0 — zero-trip: lowerBound < upperBound is false.
    %c8i = arith.constant 8 : index
    %c0i = arith.constant 0 : index
    %c1i = arith.constant 1 : index
    %result = scf.for %i = %c8i to %c0i step %c1i iter_args(%acc = %c99) -> i32 {
      // Never executed.
      %bad = arith.constant 0 : i32
      scf.yield %bad : i32
    }
    // The init value (99) flows through as the loop result.
    ptr.store %result, %buf : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    return
  }
}
