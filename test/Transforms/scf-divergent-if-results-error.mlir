// RUN: not aster-opt %s --pass-pipeline='builtin.module(aster-convert-scf-control-flow)' --split-input-file 2>&1 | FileCheck %s

// Verify that a divergent scf.if with an else branch and SSA results is
// rejected with a proper error, since the lowering cannot handle per-lane
// result merging on the divergent path.

// CHECK: divergent scf.if with results is not supported

amdgcn.module @test_mod target = <gfx942> {

kernel @test_divergent_if_with_results {
  %tid = aster_utils.thread_id x
  %c0 = arith.constant 0 : i32
  // arith.cmpi on a per-lane value produces a divergent condition.
  %cond = arith.cmpi slt, %tid, %c0 : i32
  // scf.if with results on a divergent condition is not supported.
  %result = scf.if %cond -> (i32) {
    %c1 = arith.constant 1 : i32
    scf.yield %c1 : i32
  } else {
    %c2 = arith.constant 2 : i32
    scf.yield %c2 : i32
  }
  end_kernel
}

} // amdgcn.module
