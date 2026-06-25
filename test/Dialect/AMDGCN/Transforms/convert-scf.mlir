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
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK-DAG:       %[[C42:.*]] = arith.constant 42 : i32
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[C42]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
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
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] : i1
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[REG1:.*]] = lsir.to_reg %[[C1]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG1]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] invert : i1
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           %[[REG2:.*]] = lsir.to_reg %[[C2]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG2]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
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
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] : i1
// CHECK:           cf.br ^bb3(%[[A]] : i32)
// CHECK:         ^bb2:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] invert : i1
// CHECK:           cf.br ^bb3(%[[B]] : i32)
// CHECK:         ^bb3(%[[RESULT:.*]]: i32):
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
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
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^[[BB_THEN:.*]], ^[[BB_MERGE:.*]]
// CHECK:         ^[[BB_THEN]]:
// CHECK:           %[[REG_THEN:.*]] = lsir.to_reg %[[IV]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG_THEN]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^[[BB_MERGE]]
// CHECK:         ^[[BB_MERGE]]:
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
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
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^[[BB_THEN:.*]], ^[[BB_IF_MERGE:.*]]
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
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
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
// CHECK:           %[[NARROW1:.*]], %[[TOK1:.*]] = aster_utils.save_cf_mask %[[COND1_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW1]], ^[[BB_OUTER_THEN:.*]], ^[[BB_OUTER_MERGE:.*]]
// CHECK:         ^[[BB_OUTER_THEN]]:
// CHECK:           %[[NARROW2:.*]], %[[TOK2:.*]] = aster_utils.save_cf_mask %[[COND2_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW2]], ^[[BB_INNER_THEN:.*]], ^[[BB_INNER_MERGE:.*]]
// CHECK:         ^[[BB_INNER_THEN]]:
// CHECK-DAG:       %[[C42:.*]] = arith.constant 42 : i32
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[C42]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^[[BB_INNER_MERGE]]
// CHECK:         ^[[BB_INNER_MERGE]]:
// CHECK:           aster_utils.restore_cf_mask %[[TOK2]] : !aster_utils.mask_token
// CHECK:           cf.br ^[[BB_OUTER_MERGE]]
// CHECK:         ^[[BB_OUTER_MERGE]]:
// CHECK:           aster_utils.restore_cf_mask %[[TOK1]] : !aster_utils.mask_token
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
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^[[BB_THEN:.*]], ^[[BB_ELSE:.*]]
// CHECK:         ^[[BB_THEN]]:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] : i1
// CHECK:           %[[THEN_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C4]] : i32
// CHECK:           cf.cond_br %[[THEN_CMP]], ^[[BB_THEN_LOOP:.*]](%[[C0]] : i32), ^[[BB_THEN_END:.*]]
// CHECK:         ^[[BB_THEN_LOOP]](%[[IV_THEN:.*]]: i32):
// CHECK:           amdgcn.test_inst
// CHECK:           cf.cond_br {{.*}}, ^[[BB_THEN_LOOP]]({{.*}}), ^[[BB_THEN_END]]
// CHECK:         ^[[BB_THEN_END]]:
// CHECK:           cf.br ^[[BB_MERGE:.*]]
// CHECK:         ^[[BB_ELSE]]:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] invert : i1
// CHECK:           %[[ELSE_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[C8]] : i32
// CHECK:           cf.cond_br %[[ELSE_CMP]], ^[[BB_ELSE_LOOP:.*]](%[[C0]] : i32), ^[[BB_ELSE_END:.*]]
// CHECK:         ^[[BB_ELSE_LOOP]](%[[IV_ELSE:.*]]: i32):
// CHECK:           amdgcn.test_inst
// CHECK:           cf.cond_br {{.*}}, ^[[BB_ELSE_LOOP]]({{.*}}), ^[[BB_ELSE_END]]
// CHECK:         ^[[BB_ELSE_END]]:
// CHECK:           cf.br ^[[BB_MERGE]]
// CHECK:         ^[[BB_MERGE]]:
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
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
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND_U]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^[[BB_THEN:.*]], ^[[BB_ELSE:.*]]
// CHECK:         ^[[BB_THEN]]:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] : i1
// CHECK:           %[[SUM:.*]] = arith.addi %[[ACC]], %[[IV]] : i32
// CHECK:           cf.br ^[[BB_IF_MERGE:.*]](%[[SUM]] : i32)
// CHECK:         ^[[BB_ELSE]]:
// CHECK:           aster_utils.set_cf_mask %[[COND_U]] invert : i1
// CHECK:           cf.br ^[[BB_IF_MERGE]](%[[ACC]] : i32)
// CHECK:         ^[[BB_IF_MERGE]](%[[NEW_ACC:.*]]: i32):
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
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

// Test: divergent scf.if — condition derived from thread_id (VGPR-based, no
// assume_uniform). save_cf_mask is inserted unconditionally; the divergence
// is detected at codegen time from the condition type (VCC vs SCC).
// CHECK-LABEL:   func.func @test_divergent_if_no_else(
// CHECK-SAME:      %[[TID:.*]]: i32) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[COND:.*]] = arith.cmpi sgt, %[[TID]], %[[C0]] : i32
// CHECK:           %[[NARROW:.*]], %[[TOK:.*]] = aster_utils.save_cf_mask %[[COND]] : (i1) -> (i1, !aster_utils.mask_token)
// CHECK:           cf.cond_br %[[NARROW]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK-DAG:       %[[C42:.*]] = arith.constant 42 : i32
// CHECK:           %[[REG:.*]] = lsir.to_reg %[[C42]] : i32 -> !amdgcn.sgpr
// CHECK:           amdgcn.test_inst ins %[[REG]] : (!amdgcn.sgpr) -> ()
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           aster_utils.restore_cf_mask %[[TOK]] : !aster_utils.mask_token
// CHECK:           return
// CHECK:         }
func.func @test_divergent_if_no_else(%tid: i32) {
  %c0 = arith.constant 0 : i32
  // Per-lane comparison: condition is divergent (tid differs across lanes).
  %cond = arith.cmpi sgt, %tid, %c0 : i32
  scf.if %cond {
    %c42 = arith.constant 42 : i32
    %reg = lsir.to_reg %c42 : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %reg : (!amdgcn.sgpr) -> ()
  }
  return
}

// -----

// Test: simple scf.while with one iteration variable.
// CHECK-LABEL:   func.func @test_while_simple(
// CHECK-NOT: scf.
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           cf.br ^[[BEFORE:.*]](%[[ARG0]] : i32)
// CHECK:         ^[[BEFORE]](%[[I:.*]]: i32):
// CHECK:           %[[COND:.*]] = arith.cmpi slt, %[[I]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[COND]], ^[[AFTER:.*]](%[[I]] : i32), ^[[END:.*]](%[[I]] : i32)
// CHECK:         ^[[AFTER]](%[[I2:.*]]: i32):
// CHECK:           %[[NEXT:.*]] = arith.addi %[[I2]], %[[C1]] : i32
// CHECK:           cf.br ^[[BEFORE]](%[[NEXT]] : i32)
// CHECK:         ^[[END]](%[[RES:.*]]: i32):
// CHECK:           return %[[RES]] : i32
// CHECK:         }
func.func @test_while_simple(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %res = scf.while (%i = %arg0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c10 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %next = arith.addi %i, %c1 : i32
    scf.yield %next : i32
  }
  return %res : i32
}

// -----

// Test: scf.while with two iteration variables; the before region forwards a
// subset of them as the loop result.
// CHECK-LABEL:   func.func @test_while_multi(
// CHECK-NOT: scf.
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32 {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C100:.*]] = arith.constant 100 : i32
// CHECK:           cf.br ^[[BEFORE:.*]](%[[ARG0]], %[[ARG1]] : i32, i32)
// CHECK:         ^[[BEFORE]](%[[I:.*]]: i32, %[[ACC:.*]]: i32):
// CHECK:           %[[COND:.*]] = arith.cmpi slt, %[[I]], %[[C100]] : i32
// CHECK:           cf.cond_br %[[COND]], ^[[AFTER:.*]](%[[ACC]] : i32), ^[[END:.*]](%[[ACC]] : i32)
// CHECK:         ^[[AFTER]](%[[ACC2:.*]]: i32):
// CHECK:           %[[NI:.*]] = arith.addi %[[ACC2]], %[[C1]] : i32
// CHECK:           cf.br ^[[BEFORE]](%[[NI]], %[[ACC2]] : i32, i32)
// CHECK:         ^[[END]](%[[RES:.*]]: i32):
// CHECK:           return %[[RES]] : i32
// CHECK:         }
func.func @test_while_multi(%arg0: i32, %arg1: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant 100 : i32
  %res = scf.while (%i = %arg0, %acc = %arg1) : (i32, i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c100 : i32
    scf.condition(%cond) %acc : i32
  } do {
  ^bb0(%acc: i32):
    %ni = arith.addi %acc, %c1 : i32
    scf.yield %ni, %acc : i32, i32
  }
  return %res : i32
}

// -----

// Test: scf.for nested inside the after region of scf.while; verifies that
// both the outer while and the inner for are fully lowered to CF.
// CHECK-LABEL:   func.func @test_while_with_for(
// CHECK-NOT: scf.
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           cf.br ^[[BEFORE:.*]](%[[ARG0]] : i32)
// CHECK:         ^[[BEFORE]](%[[I:.*]]: i32):
// CHECK:           %[[COND:.*]] = arith.cmpi slt, %[[I]], %[[C10]] : i32
// CHECK:           cf.cond_br %[[COND]], ^[[AFTER:.*]](%[[I]] : i32), ^[[END:.*]](%[[I]] : i32)
// CHECK:         ^[[AFTER]](%[[I2:.*]]: i32):
// CHECK:           %[[INIT_CMP:.*]] = arith.cmpi slt, %[[C0]], %[[I2]] : i32
// CHECK:           cf.cond_br %[[INIT_CMP]], ^[[FOR_BODY:.*]](%[[C0]], %[[I2]] : i32, i32), ^[[FOR_END:.*]](%[[I2]] : i32)
// CHECK:         ^[[FOR_BODY]](%[[J:.*]]: i32, %[[FACC:.*]]: i32):
// CHECK:           %[[NEXT_ACC:.*]] = arith.addi %[[FACC]], %[[J]] : i32
// CHECK:           %[[J_NEXT:.*]] = arith.addi %[[J]], %[[C1]] : i32
// CHECK:           %[[BACK_CMP:.*]] = arith.cmpi slt, %[[J_NEXT]], %[[I2]] : i32
// CHECK:           cf.cond_br %[[BACK_CMP]], ^[[FOR_BODY]](%[[J_NEXT]], %[[NEXT_ACC]] : i32, i32), ^[[FOR_END]](%[[NEXT_ACC]] : i32)
// CHECK:         ^[[FOR_END]](%[[SUM:.*]]: i32):
// CHECK:           %[[NEXT:.*]] = arith.addi %[[SUM]], %[[C1]] : i32
// CHECK:           cf.br ^[[BEFORE]](%[[NEXT]] : i32)
// CHECK:         ^[[END]](%[[RES:.*]]: i32):
// CHECK:           return %[[RES]] : i32
// CHECK:         }
func.func @test_while_with_for(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %res = scf.while (%i = %arg0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c10 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %sum = scf.for %j = %c0 to %i step %c1 iter_args(%acc = %i) -> i32 : i32 {
      %next_acc = arith.addi %acc, %j : i32
      scf.yield %next_acc : i32
    }
    %next = arith.addi %sum, %c1 : i32
    scf.yield %next : i32
  }
  return %res : i32
}

// -----

// Test: scf.while with zero inits and zero results; exercises the empty
// conditionArgs and empty bbEnd block-arg paths.
// CHECK-LABEL:   func.func @test_while_zero_results(
// CHECK-NOT: scf.
// CHECK-SAME:      %[[N:.*]]: i32) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           cf.br ^[[BEFORE:.*]]
// CHECK:         ^[[BEFORE]]:
// CHECK:           %[[COND:.*]] = arith.cmpi slt, %[[C0]], %[[N]] : i32
// CHECK:           cf.cond_br %[[COND]], ^[[AFTER:.*]], ^[[END:.*]]
// CHECK:         ^[[AFTER]]:
// CHECK:           cf.br ^[[BEFORE]]
// CHECK:         ^[[END]]:
// CHECK:           return
// CHECK:         }
func.func @test_while_zero_results(%n: i32) {
  %c0 = arith.constant 0 : i32
  scf.while () : () -> () {
    %cond = arith.cmpi slt, %c0, %n : i32
    scf.condition(%cond)
  } do {
    scf.yield
  }
  return
}

// -----

// Test: scf.while where the forwarded condition value is an op result (not a
// block arg), exercising IRMapping remap of a computed value.
// CHECK-LABEL:   func.func @test_while_computed_forward(
// CHECK-NOT: scf.
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[C50:.*]] = arith.constant 50 : i32
// CHECK:           cf.br ^[[BEFORE:.*]](%[[ARG0]] : i32)
// CHECK:         ^[[BEFORE]](%[[I:.*]]: i32):
// CHECK:           %[[DOUBLED:.*]] = arith.muli %[[I]], %[[C1]] : i32
// CHECK:           %[[COND:.*]] = arith.cmpi slt, %[[I]], %[[C50]] : i32
// CHECK:           cf.cond_br %[[COND]], ^[[AFTER:.*]](%[[DOUBLED]] : i32), ^[[END:.*]](%[[DOUBLED]] : i32)
// CHECK:         ^[[AFTER]](%[[V:.*]]: i32):
// CHECK:           %[[NEXT:.*]] = arith.addi %[[V]], %[[C1]] : i32
// CHECK:           cf.br ^[[BEFORE]](%[[NEXT]] : i32)
// CHECK:         ^[[END]](%[[RES:.*]]: i32):
// CHECK:           return %[[RES]] : i32
// CHECK:         }
func.func @test_while_computed_forward(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c50 = arith.constant 50 : i32
  %res = scf.while (%i = %arg0) : (i32) -> i32 {
    %doubled = arith.muli %i, %c1 : i32
    %cond = arith.cmpi slt, %i, %c50 : i32
    scf.condition(%cond) %doubled : i32
  } do {
  ^bb0(%v: i32):
    %next = arith.addi %v, %c1 : i32
    scf.yield %next : i32
  }
  return %res : i32
}
