// RUN: aster-opt %s --aster-convert-scf-control-flow --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @test_uniform_loops_const_bounds() {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^bb1(%[[C0]] : i32), ^bb2
// CHECK:         ^bb1(%[[IV:.*]]: i32):
// CHECK:           %[[TO_REG:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[TO_REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^bb1(%[[IV_NEXT]] : i32), ^bb2
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
func.func @test_uniform_loops_const_bounds() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  scf.for %i = %c0 to %c10 step %c1 : i32 {
    %iv = lsir.to_reg %i : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %iv : (!amdgcn.sgpr) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @test_uniform_loops_non_const_bounds(
// CHECK-SAME:      %[[ARG0:.*]]: i32) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[N_U:.*]] = aster_utils.assume_uniform %[[ARG0]] : i32
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[N_U]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^bb1(%[[C0]] : i32), ^bb2
// CHECK:         ^bb1(%[[IV:.*]]: i32):
// CHECK:           %[[TO_REG:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[TO_REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[N_U]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^bb1(%[[IV_NEXT]] : i32), ^bb2
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
func.func @test_uniform_loops_non_const_bounds(%n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %n_u = aster_utils.assume_uniform %n : i32
  scf.for %i = %c0 to %n_u step %c1 : i32 {
    %iv = lsir.to_reg %i : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %iv : (!amdgcn.sgpr) -> ()
  }
  return
}

// -----

// CHECK-LABEL:   func.func @test_uniform_if_no_else(
// CHECK-SAME:      %[[COND:.*]]: i1) {
// CHECK:           %[[COND_U:.*]] = aster_utils.assume_uniform %[[COND]] : i1
// CHECK:           cf.cond_br %[[COND_U]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK-DAG:       %[[C42:.*]] = arith.constant 42 : i32
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[C42]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
func.func @test_uniform_if_no_else(%cond: i1) {
  %cond_u = aster_utils.assume_uniform %cond : i1
  scf.if %cond_u {
    %c42 = arith.constant 42 : i32
    %reg = lsir.to_reg %c42 : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @test_uniform_if_else_no_results(
// CHECK-SAME:      %[[COND:.*]]: i1) {
// CHECK:           %[[COND_U:.*]] = aster_utils.assume_uniform %[[COND]] : i1
// CHECK:           cf.cond_br %[[COND_U]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[REG1:.*]] = lsir.to_reg %[[C1]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG1]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           %[[REG2:.*]] = lsir.to_reg %[[C2]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG2]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return
// CHECK:         }
func.func @test_uniform_if_else_no_results(%cond: i1) {
  %cond_u = aster_utils.assume_uniform %cond : i1
  scf.if %cond_u {
    %c1 = arith.constant 1 : i32
    %reg1 = lsir.to_reg %c1 : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg1 : (!amdgcn.sgpr) -> ()
  } else {
    %c2 = arith.constant 2 : i32
    %reg2 = lsir.to_reg %c2 : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg2 : (!amdgcn.sgpr) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @test_uniform_if_else_with_results(
// CHECK-SAME:      %[[COND:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32) {
// CHECK:           %[[COND_U:.*]] = aster_utils.assume_uniform %[[COND]] : i1
// CHECK:           cf.cond_br %[[COND_U]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3(%[[A]] : i32)
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3(%[[B]] : i32)
// CHECK:         ^bb3(%[[RESULT:.*]]: i32):
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[RESULT]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_uniform_if_else_with_results(%cond: i1, %a: i32, %b: i32) {
  %cond_u = aster_utils.assume_uniform %cond : i1
  %result = scf.if %cond_u -> i32 {
    scf.yield %a : i32
  } else {
    scf.yield %b : i32
  }
  %reg = lsir.to_reg %result : i32 -> !amdgcn.sgpr
  amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
  return
}

// -----

// Test: scf.if nested inside scf.for
// CHECK-LABEL:   func.func @test_if_inside_for(
// CHECK-SAME:      %[[COND:.*]]: i1) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           %[[COND_U:.*]] = aster_utils.assume_uniform %[[COND]] : i1
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^[[BB_BODY:.*]](%[[C0]] : i32), ^[[BB_END:.*]]
// CHECK:         ^[[BB_BODY]](%[[IV:.*]]: i32):
// CHECK:           cf.cond_br %[[COND_U]], ^[[BB_THEN:.*]], ^[[BB_MERGE:.*]]
// CHECK:         ^[[BB_THEN]]:
// CHECK:           %[[REG_THEN:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG_THEN]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^[[BB_MERGE]]
// CHECK:         ^[[BB_MERGE]]:
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^[[BB_BODY]](%[[IV_NEXT]] : i32), ^[[BB_END]]
// CHECK:         ^[[BB_END]]:
// CHECK:           return
// CHECK:         }
func.func @test_if_inside_for(%cond: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %cond_u = aster_utils.assume_uniform %cond : i1
  scf.for %i = %c0 to %c10 step %c1 : i32 {
    scf.if %cond_u {
      %reg = lsir.to_reg %i : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
  }
  return
}

// -----

// Test: scf.for nested inside scf.if (then branch)
// CHECK-LABEL:   func.func @test_for_inside_if(
// CHECK-SAME:      %[[COND:.*]]: i1) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C5:.*]] = arith.constant 5 : i32
// CHECK:           %[[COND_U:.*]] = aster_utils.assume_uniform %[[COND]] : i1
// CHECK:           cf.cond_br %[[COND_U]], ^[[BB_THEN:.*]], ^[[BB_IF_MERGE:.*]]
// CHECK:         ^[[BB_THEN]]:
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C5]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^[[BB_LOOP:.*]](%[[C0]] : i32), ^[[BB_LOOP_END:.*]]
// CHECK:         ^[[BB_LOOP]](%[[IV:.*]]: i32):
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[C5]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^[[BB_LOOP]](%[[IV_NEXT]] : i32), ^[[BB_LOOP_END]]
// CHECK:         ^[[BB_LOOP_END]]:
// CHECK:           cf.br ^[[BB_IF_MERGE]]
// CHECK:         ^[[BB_IF_MERGE]]:
// CHECK:           return
// CHECK:         }
func.func @test_for_inside_if(%cond: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c5 = arith.constant 5 : i32
  %cond_u = aster_utils.assume_uniform %cond : i1
  scf.if %cond_u {
    scf.for %i = %c0 to %c5 step %c1 : i32 {
      %reg = lsir.to_reg %i : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
  }
  return
}

// -----

// Test: scf.if nested inside scf.if
// CHECK-LABEL:   func.func @test_if_inside_if(
// CHECK-SAME:      %[[COND1:.*]]: i1, %[[COND2:.*]]: i1) {
// CHECK:           %[[COND1_U:.*]] = aster_utils.assume_uniform %[[COND1]] : i1
// CHECK:           %[[COND2_U:.*]] = aster_utils.assume_uniform %[[COND2]] : i1
// CHECK:           cf.cond_br %[[COND1_U]], ^[[BB_OUTER_THEN:.*]], ^[[BB_OUTER_MERGE:.*]]
// CHECK:         ^[[BB_OUTER_THEN]]:
// CHECK:           cf.cond_br %[[COND2_U]], ^[[BB_INNER_THEN:.*]], ^[[BB_INNER_MERGE:.*]]
// CHECK:         ^[[BB_INNER_THEN]]:
// CHECK-DAG:       %[[C42:.*]] = arith.constant 42 : i32
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[C42]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^[[BB_INNER_MERGE]]
// CHECK:         ^[[BB_INNER_MERGE]]:
// CHECK:           cf.br ^[[BB_OUTER_MERGE]]
// CHECK:         ^[[BB_OUTER_MERGE]]:
// CHECK:           return
// CHECK:         }
func.func @test_if_inside_if(%cond1: i1, %cond2: i1) {
  %cond1_u = aster_utils.assume_uniform %cond1 : i1
  %cond2_u = aster_utils.assume_uniform %cond2 : i1
  scf.if %cond1_u {
    scf.if %cond2_u {
      %c42 = arith.constant 42 : i32
      %reg = lsir.to_reg %c42 : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
  }
  return
}

// -----

// Test: scf.for nested inside scf.for
// CHECK-LABEL:   func.func @test_for_inside_for() {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : i32
// CHECK:           %[[INIT_CMP_OUTER:.*]] = arith.cmpi slt, %[[C0]], %[[C4]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP_OUTER]], ^[[BB_OUTER:.*]](%[[C0]] : i32), ^[[BB_EXIT:.*]]
// CHECK:         ^[[BB_OUTER]](%[[IV_OUTER:.*]]: i32):
// CHECK:           %[[INIT_CMP_INNER:.*]] = arith.cmpi slt, %[[C0]], %[[C8]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP_INNER]], ^[[BB_INNER:.*]](%[[C0]] : i32), ^[[BB_INNER_END:.*]]
// CHECK:         ^[[BB_INNER]](%[[IV_INNER:.*]]: i32):
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[IV_INNER]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[IV_INNER_NEXT:.*]] = arith.addi %[[IV_INNER]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP_INNER:.*]] = arith.cmpi slt, %[[IV_INNER_NEXT]], %[[C8]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP_INNER]], ^[[BB_INNER]](%[[IV_INNER_NEXT]] : i32), ^[[BB_INNER_END]]
// CHECK:         ^[[BB_INNER_END]]:
// CHECK:           %[[IV_OUTER_NEXT:.*]] = arith.addi %[[IV_OUTER]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP_OUTER:.*]] = arith.cmpi slt, %[[IV_OUTER_NEXT]], %[[C4]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP_OUTER]], ^[[BB_OUTER]](%[[IV_OUTER_NEXT]] : i32), ^[[BB_EXIT]]
// CHECK:         ^[[BB_EXIT]]:
// CHECK:           return
// CHECK:         }
func.func @test_for_inside_for() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %c4 step %c1 : i32 {
    scf.for %j = %c0 to %c8 step %c1 : i32 {
      %reg = lsir.to_reg %j : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
  }
  return
}

// -----

// Test: scf.if with else, both branches containing scf.for
// CHECK-LABEL:   func.func @test_for_in_both_if_branches(
// CHECK-SAME:      %[[COND:.*]]: i1) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : i32
// CHECK:           %[[COND_U:.*]] = aster_utils.assume_uniform %[[COND]] : i1
// CHECK:           cf.cond_br %[[COND_U]], ^[[BB_THEN:.*]], ^[[BB_ELSE:.*]]
// CHECK:         ^[[BB_THEN]]:
// CHECK:           %[[THEN_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C4]] : i32
// CHECK:           cf.cond_br %[[THEN_CMP]], ^[[BB_THEN_LOOP:.*]](%[[C0]] : i32), ^[[BB_THEN_END:.*]]
// CHECK:         ^[[BB_THEN_LOOP]](%[[IV_THEN:.*]]: i32):
// CHECK:           amdgcn.test_inst
// CHECK:           cf.cond_br {{.*}}, ^[[BB_THEN_LOOP]]({{.*}}), ^[[BB_THEN_END]]
// CHECK:         ^[[BB_THEN_END]]:
// CHECK:           cf.br ^[[BB_MERGE:.*]]
// CHECK:         ^[[BB_ELSE]]:
// CHECK:           %[[ELSE_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C8]] : i32
// CHECK:           cf.cond_br %[[ELSE_CMP]], ^[[BB_ELSE_LOOP:.*]](%[[C0]] : i32), ^[[BB_ELSE_END:.*]]
// CHECK:         ^[[BB_ELSE_LOOP]](%[[IV_ELSE:.*]]: i32):
// CHECK:           amdgcn.test_inst
// CHECK:           cf.cond_br {{.*}}, ^[[BB_ELSE_LOOP]]({{.*}}), ^[[BB_ELSE_END]]
// CHECK:         ^[[BB_ELSE_END]]:
// CHECK:           cf.br ^[[BB_MERGE]]
// CHECK:         ^[[BB_MERGE]]:
// CHECK:           return
// CHECK:         }
func.func @test_for_in_both_if_branches(%cond: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %cond_u = aster_utils.assume_uniform %cond : i1
  scf.if %cond_u {
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %reg = lsir.to_reg %i : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
  } else {
    scf.for %j = %c0 to %c8 step %c1 : i32 {
      %reg = lsir.to_reg %j : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
  }
  return
}

// -----

// Test: scf.if with results nested inside scf.for (iter_args)
// CHECK-LABEL:   func.func @test_if_with_results_inside_for(
// CHECK-SAME:      %[[COND:.*]]: i1, %[[INIT:.*]]: i32) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           %[[COND_U:.*]] = aster_utils.assume_uniform %[[COND]] : i1
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^[[BB_BODY:.*]](%[[C0]], %[[INIT]] : i32, i32), ^[[BB_EXIT:.*]](%[[INIT]] : i32)
// CHECK:         ^[[BB_BODY]](%[[IV:.*]]: i32, %[[ACC:.*]]: i32):
// CHECK:           cf.cond_br %[[COND_U]], ^[[BB_THEN:.*]], ^[[BB_ELSE:.*]]
// CHECK:         ^[[BB_THEN]]:
// CHECK:           %[[SUM:.*]] = arith.addi %[[ACC]], %[[IV]] : i32
// CHECK:           cf.br ^[[BB_IF_MERGE:.*]](%[[SUM]] : i32)
// CHECK:         ^[[BB_ELSE]]:
// CHECK:           cf.br ^[[BB_IF_MERGE]](%[[ACC]] : i32)
// CHECK:         ^[[BB_IF_MERGE]](%[[NEW_ACC:.*]]: i32):
// CHECK:           %[[IV_NEXT:.*]] = arith.addi %[[IV]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^[[BB_BODY]](%[[IV_NEXT]], %[[NEW_ACC]] : i32, i32), ^[[BB_EXIT]](%[[NEW_ACC]] : i32)
// CHECK:         ^[[BB_EXIT]](%[[RESULT:.*]]: i32):
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[RESULT]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_if_with_results_inside_for(%cond: i1, %init: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %cond_u = aster_utils.assume_uniform %cond : i1
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> i32 : i32 {
    %new_acc = scf.if %cond_u -> i32 {
      %sum = arith.addi %acc, %i : i32
      scf.yield %sum : i32
    } else {
      scf.yield %acc : i32
    }
    scf.yield %new_acc : i32
  }
  %reg = lsir.to_reg %result : i32 -> !amdgcn.sgpr
  amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
  return
}

// -----

// Test: uniform scf.while — no mask ops should be emitted.
// CHECK-LABEL:   func.func @test_uniform_while(
// CHECK-SAME:      %[[N:.*]]: i32) {
// CHECK-NOT:       aster_utils.get_cf_mask
// CHECK-NOT:       aster_utils.set_cf_mask
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[N_U:.*]] = aster_utils.assume_uniform %[[N]] : i32
// CHECK:           cf.br ^[[BB_BEFORE:.*]](%[[C0]] : i32)
// CHECK:         ^[[BB_BEFORE]](%[[B0:.*]]: i32):
// CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[B0]], %[[N_U]] : i32
// CHECK:           cf.cond_br %[[CMP]], ^[[BB_AFTER:.*]](%[[B0]] : i32), ^[[BB_END:.*]](%[[B0]] : i32)
// CHECK:         ^[[BB_AFTER]](%[[A0:.*]]: i32):
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[A0]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           %[[NEXT:.*]] = arith.addi %[[A0]], %[[C1]] : i32
// CHECK:           cf.br ^[[BB_BEFORE]](%[[NEXT]] : i32)
// CHECK:         ^[[BB_END]](%[[RES:.*]]: i32):
// CHECK:           %[[OUT:.*]] = lsir.to_reg %[[RES]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[OUT]] : (!amdgcn.sgpr) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_uniform_while(%n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %n_u = aster_utils.assume_uniform %n : i32
  %result = scf.while (%i = %c0) : (i32) -> i32 {
    %cmp = arith.cmpi slt, %i, %n_u : i32
    scf.condition(%cmp) %i : i32
  } do {
  ^bb0(%a: i32):
    %reg = lsir.to_reg %a : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    %next = arith.addi %a, %c1 : i32
    scf.yield %next : i32
  }
  %out = lsir.to_reg %result : i32 -> !amdgcn.sgpr
  amdgcn.test_inst ins %out : (!amdgcn.sgpr) -> ()
  return
}

// -----

// Test: divergent scf.while with one loop-carried value — mask save/restore
// must be emitted.
// CHECK-LABEL:   func.func @test_divergent_while_single_iter_arg(
// CHECK-SAME:      %[[INIT:.*]]: i32) {
// CHECK:           %[[SAVED:.*]] = aster_utils.get_cf_mask : i64
// CHECK:           cf.br ^[[BB_BEFORE:.*]](%[[INIT]] : i32)
// CHECK:         ^[[BB_BEFORE]](%[[B0:.*]]: i32):
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:           %[[TID:.*]] = aster_utils.thread_id x
// CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[TID]], %[[B0]] : i32
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]], %[[CMP]] : i64
// CHECK:           cf.cond_br %[[CMP]], ^[[BB_AFTER:.*]](%[[B0]] : i32), ^[[BB_END:.*]](%[[B0]] : i32)
// CHECK:         ^[[BB_AFTER]](%[[A0:.*]]: i32):
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[A0]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^[[BB_BEFORE]](%[[A0]] : i32)
// CHECK:         ^[[BB_END]](%[[RES:.*]]: i32):
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:           %[[OUT:.*]] = lsir.to_reg %[[RES]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[OUT]] : (!amdgcn.sgpr) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_divergent_while_single_iter_arg(%init: i32) {
  %result = scf.while (%val = %init) : (i32) -> i32 {
    %tid = aster_utils.thread_id x
    %cmp = arith.cmpi slt, %tid, %val : i32
    scf.condition(%cmp) %val : i32
  } do {
  ^bb0(%a: i32):
    %reg = lsir.to_reg %a : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    scf.yield %a : i32
  }
  %out = lsir.to_reg %result : i32 -> !amdgcn.sgpr
  amdgcn.test_inst ins %out : (!amdgcn.sgpr) -> ()
  return
}

// -----

// Test: divergent scf.while with two loop-carried values. Both must appear as
// block arguments on the before block; only the forwarded subset appears on
// the after block and the end block.
// CHECK-LABEL:   func.func @test_divergent_while_iter_args(
// CHECK-SAME:      %[[A:.*]]: i32, %[[B:.*]]: i32) {
// CHECK:           %[[SAVED:.*]] = aster_utils.get_cf_mask : i64
// CHECK:           cf.br ^[[BB_BEFORE:.*]](%[[A]], %[[B]] : i32, i32)
// CHECK:         ^[[BB_BEFORE]](%[[B0:.*]]: i32, %[[B1:.*]]: i32):
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:           %[[TID:.*]] = aster_utils.thread_id x
// CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[TID]], %[[B0]] : i32
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]], %[[CMP]] : i64
// CHECK:           cf.cond_br %[[CMP]], ^[[BB_AFTER:.*]](%[[B1]] : i32), ^[[BB_END:.*]](%[[B1]] : i32)
// CHECK:         ^[[BB_AFTER]](%[[A0:.*]]: i32):
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[A0]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^[[BB_BEFORE]](%[[A0]], %[[A0]] : i32, i32)
// CHECK:         ^[[BB_END]](%[[RES:.*]]: i32):
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:           %[[OUT:.*]] = lsir.to_reg %[[RES]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[OUT]] : (!amdgcn.sgpr) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_divergent_while_iter_args(%a: i32, %b: i32) {
  // The before region has two args (%x, %y); only %y is forwarded as the
  // result. The after region uses %acc (= %y) and yields (%acc, %acc) so
  // both next before-args are derived solely from the after-region argument.
  %result = scf.while (%x = %a, %y = %b) : (i32, i32) -> i32 {
    %tid = aster_utils.thread_id x
    %cmp = arith.cmpi slt, %tid, %x : i32
    scf.condition(%cmp) %y : i32
  } do {
  ^bb0(%acc: i32):
    %reg = lsir.to_reg %acc : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    scf.yield %acc, %acc : i32, i32
  }
  %out = lsir.to_reg %result : i32 -> !amdgcn.sgpr
  amdgcn.test_inst ins %out : (!amdgcn.sgpr) -> ()
  return
}

// -----

// Test: divergent scf.while where the condition may be false on the first
// iteration. The false edge of cf.cond_br must target the end block with the
// forwarded values, confirming the exit-on-first-check path is structurally
// identical to the general divergent case.
// CHECK-LABEL:   func.func @test_divergent_while_first_iter_exit(
// CHECK-SAME:      %[[INIT:.*]]: i32) {
// CHECK:           %[[SAVED:.*]] = aster_utils.get_cf_mask : i64
// CHECK:           cf.br ^[[BB_BEFORE:.*]](%[[INIT]] : i32)
// CHECK:         ^[[BB_BEFORE]](%[[B0:.*]]: i32):
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:           %[[TID:.*]] = aster_utils.thread_id x
// CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[TID]], %[[B0]] : i32
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]], %[[CMP]] : i64
// CHECK:           cf.cond_br %[[CMP]], ^[[BB_AFTER:.*]](%[[B0]] : i32), ^[[BB_END:.*]](%[[B0]] : i32)
// CHECK:         ^[[BB_AFTER]](%[[A0:.*]]: i32):
// CHECK:           cf.br ^[[BB_BEFORE]](%[[A0]] : i32)
// CHECK:         ^[[BB_END]](%[[RES:.*]]: i32):
// CHECK-NEXT:      aster_utils.set_cf_mask %[[SAVED]] : i64
// CHECK:           return
// CHECK:         }
func.func @test_divergent_while_first_iter_exit(%init: i32) {
  // The condition can be false immediately, exercising the exit edge without
  // ever entering the after region.
  %result = scf.while (%val = %init) : (i32) -> i32 {
    %tid = aster_utils.thread_id x
    %cmp = arith.cmpi slt, %tid, %val : i32
    scf.condition(%cmp) %val : i32
  } do {
  ^bb0(%a: i32):
    scf.yield %a : i32
  }
  return
}

// -----

// Divergent scf.for: lower bound is thread_id, so threads enter/exit at
// different iterations. EXEC is saved before the loop and the active-lane
// condition is passed as a block argument into bbBody so the mask is narrowed
// only at the top of each iteration body and restored once after the loop.
// No set_cf_mask is emitted before the initial branch.
// CHECK-LABEL: kernel @k_divergent_for
// CHECK:         %[[SAVED:.*]] = aster_utils.get_cf_mask : i64
// CHECK:         %[[INIT_CMP:.*]] = arith.cmpi slt, %[[TID:.*]], %{{.*}} : i32
// CHECK-NEXT:    cf.cond_br %[[INIT_CMP]], ^[[BB_BODY:bb[0-9]+]](%[[TID]], %[[INIT_CMP]] : i32, i1), ^[[BB_END:bb[0-9]+]]
// CHECK:       ^[[BB_BODY]](%[[IV:.*]]: i32, %[[COND:.*]]: i1):
// CHECK-NEXT:    aster_utils.set_cf_mask %[[SAVED]], %[[COND]] : i64
// CHECK-NEXT:    %[[REG:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK-NEXT:    test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK-NEXT:    %[[IV_NEXT:.*]] = arith.addi %[[IV]], %{{.*}} : i32
// CHECK-NEXT:    %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %{{.*}} : i32
// CHECK-NEXT:    cf.cond_br %[[BACK_CMP]], ^[[BB_BODY]](%[[IV_NEXT]], %[[BACK_CMP]] : i32, i1), ^[[BB_END]]
// CHECK:       ^[[BB_END]]:
// CHECK-NEXT:    aster_utils.set_cf_mask %[[SAVED]] : i64
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_divergent_for {
    %tid = aster_utils.thread_id x
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    scf.for %i = %tid to %c10 step %c1 : i32 {
      %reg = lsir.to_reg %i : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
    end_kernel
  }
}

// -----

// Uniform scf.for inside a kernel: no mask ops despite a wave target present.
// CHECK-LABEL: kernel @k_uniform_for
// CHECK-NOT:     aster_utils.get_cf_mask
// CHECK-NOT:     aster_utils.set_cf_mask
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_uniform_for {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    scf.for %i = %c0 to %c10 step %c1 : i32 {
      %reg = lsir.to_reg %i : i32 -> !amdgcn.sgpr
      amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    }
    end_kernel
  }
}

// -----

// Divergent scf.for with one iter_arg: the active-lane condition is a bbarg
// alongside iv and the iter_arg; mask narrowing happens only at the top of
// each body iteration; the mask is restored once after the loop.
// CHECK-LABEL: kernel @k_divergent_for_iter_args
// CHECK:         %[[SAVED:.*]] = aster_utils.get_cf_mask : i64
// CHECK:         %[[INIT_CMP:.*]] = arith.cmpi slt, %[[TID:.*]], %{{.*}} : i32
// CHECK-NEXT:    cf.cond_br %[[INIT_CMP]], ^[[BB_BODY:bb[0-9]+]](%[[TID]], %[[INIT_CMP]], %{{.*}} : i32, i1, i32), ^[[BB_END:bb[0-9]+]](%{{.*}} : i32)
// CHECK:       ^[[BB_BODY]](%[[IV:.*]]: i32, %[[COND:.*]]: i1, %[[ACC:.*]]: i32):
// CHECK-NEXT:    aster_utils.set_cf_mask %[[SAVED]], %[[COND]] : i64
// CHECK-NEXT:    %[[SUM:.*]] = arith.addi %[[ACC]], %[[IV]] : i32
// CHECK-NEXT:    %[[IV_NEXT:.*]] = arith.addi %[[IV]], %{{.*}} : i32
// CHECK-NEXT:    %[[BACK_CMP:.*]] = arith.cmpi slt, %[[IV_NEXT]], %{{.*}} : i32
// CHECK-NEXT:    cf.cond_br %[[BACK_CMP]], ^[[BB_BODY]](%[[IV_NEXT]], %[[BACK_CMP]], %[[SUM]] : i32, i1, i32), ^[[BB_END]](%[[SUM]] : i32)
// CHECK:       ^[[BB_END]](%[[RESULT:.*]]: i32):
// CHECK-NEXT:    aster_utils.set_cf_mask %[[SAVED]] : i64
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_divergent_for_iter_args {
    %tid = aster_utils.thread_id x
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %c0 = arith.constant 0 : i32
    %result = scf.for %i = %tid to %c10 step %c1 iter_args(%acc = %c0) -> i32 : i32 {
      %sum = arith.addi %acc, %i : i32
      scf.yield %sum : i32
    }
    %reg = lsir.to_reg %result : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}
