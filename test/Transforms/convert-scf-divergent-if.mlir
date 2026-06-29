// RUN: aster-opt %s --aster-convert-scf-control-flow --split-input-file --verify-diagnostics | FileCheck %s

// -----

// Divergent scf.if with no else: get_cf_mask saves EXEC, set_cf_mask narrows
// it to condition lanes at the start of then-block, and restores at the end.

// CHECK-LABEL: kernel @k_divergent_no_else
// CHECK:         %[[SAVED:.*]] = aster_utils.get_cf_mask : i64
// CHECK-NEXT:    aster_utils.set_cf_mask %[[SAVED]], %[[COND:.*]] : i64
// CHECK-NEXT:    cf.cond_br %[[COND]], ^[[BB1:bb[0-9]+]], ^[[BB2:bb[0-9]+]]
// CHECK:       ^[[BB1]]:
// CHECK:         aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:         cf.br ^[[BB2]]
// CHECK:       ^[[BB2]]:
// CHECK-NOT:     aster_utils.get_cf_mask
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_divergent_no_else {
    %tid = aster_utils.thread_id x
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %tid, %c0 : i32
    scf.if %cond {
      %c1 = arith.constant 1 : i32
      %reg = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
    end_kernel
  }
}

// -----

// Divergent scf.if with else, no results: the then-block narrows to condition
// lanes, then restores; the else-block narrows to complement lanes, then
// restores at the merge.

// CHECK-LABEL: kernel @k_divergent_else
// CHECK:         %[[SAVED:.*]] = aster_utils.get_cf_mask : i64
// CHECK-NEXT:    aster_utils.set_cf_mask %[[SAVED]], %[[COND:.*]] : i64
// CHECK-NEXT:    cf.cond_br %[[COND]], ^[[BB1:bb[0-9]+]], ^[[BB2:bb[0-9]+]]
// CHECK:       ^[[BB1]]:
// CHECK:         aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:         cf.br ^[[BB2]]
// CHECK:       ^[[BB2]]:
// CHECK-NEXT:    aster_utils.set_cf_mask %[[SAVED]], %[[COND]]{complement} : i64
// CHECK:         aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:         cf.br ^[[BB3:bb[0-9]+]]
// CHECK:       ^[[BB3]]:
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_divergent_else {
    %tid = aster_utils.thread_id x
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %tid, %c0 : i32
    scf.if %cond {
      %c1 = arith.constant 1 : i32
      %reg = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    } else {
      %c2 = arith.constant 2 : i32
      %reg = lsir.to_reg %c2 : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
    end_kernel
  }
}

// -----

// Uniform scf.if: no get_cf_mask or set_cf_mask should appear.

// CHECK-LABEL: kernel @k_uniform
// CHECK-NOT:     aster_utils.get_cf_mask
// CHECK-NOT:     aster_utils.set_cf_mask
// CHECK:         cf.cond_br
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_uniform {
    %c1 = arith.constant 1 : i1
    %cond = aster_utils.assume_uniform %c1 : i1
    scf.if %cond {
      %c42 = arith.constant 42 : i32
      %reg = lsir.to_reg %c42 : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
    end_kernel
  }
}

// -----

// Divergent scf.if with results must be rejected with an error.

amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_divergent_results {
    %tid = aster_utils.thread_id x
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %tid, %c0 : i32
    // expected-error @+1 {{divergent scf.if with results is not supported}}
    %result = scf.if %cond -> (i32) {
      %c1 = arith.constant 1 : i32
      scf.yield %c1 : i32
    } else {
      %c2 = arith.constant 2 : i32
      scf.yield %c2 : i32
    }
    end_kernel
  }
}
