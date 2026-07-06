// RUN: aster-opt %s --amdgcn-hazards --split-input-file | FileCheck %s

// Case 21: TransOpHazard -- a trans op (v_sqrt_f32, v_rcp_f32, etc.) writing
// a VGPR followed by a non-trans VALU op reading that VGPR requires 1 V_NOP.

// CHECK-LABEL:   func.func @trans_op_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<2>) {
// CHECK:           amdgcn.v_sqrt_f32 outs(%[[ARG0]]) ins(%[[ARG1]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_add_f32 outs(%[[ARG2]]) ins(%[[ARG0]], %[[ARG1]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @trans_op_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<2>) {
  amdgcn.v_sqrt_f32 outs(%arg0) ins(%arg1) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_add_f32 outs(%arg2) ins(%arg0, %arg1) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
  return
}

// -----

// An independent VALU op separating the trans op from its consumer satisfies
// the wait state; no additional v_nop should be inserted.
// CHECK-LABEL:   func.func @trans_op_no_hazard_with_separator(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<2>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<3>) {
// CHECK:           amdgcn.v_sqrt_f32 outs(%[[ARG0]]) ins(%[[ARG1]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG3]]) ins(%[[ARG1]]) : outs(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<1>)
// CHECK-NOT:       amdgcn.v_nop
// CHECK:           amdgcn.v_add_f32 outs(%[[ARG2]]) ins(%[[ARG0]], %[[ARG1]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @trans_op_no_hazard_with_separator(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<2>, %arg3: !amdgcn.vgpr<3>) {
  amdgcn.v_sqrt_f32 outs(%arg0) ins(%arg1) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  // Intervening non-trans VALU satisfies the 1 wait state.
  amdgcn.v_mov_b32 outs(%arg3) ins(%arg1) : outs(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_add_f32 outs(%arg2) ins(%arg0, %arg1) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
  return
}

// -----

// The trans op's output VGPR is not read by the following VALU op; no hazard.
// CHECK-LABEL:   func.func @trans_op_no_hazard_different_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<2>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<3>) {
// CHECK:           amdgcn.v_sqrt_f32 outs(%[[ARG0]]) ins(%[[ARG1]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK-NOT:       amdgcn.v_nop
// CHECK:           amdgcn.v_add_f32 outs(%[[ARG2]]) ins(%[[ARG1]], %[[ARG3]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<3>)
// CHECK:           return
// CHECK:         }
func.func @trans_op_no_hazard_different_vgpr(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<2>, %arg3: !amdgcn.vgpr<3>) {
  amdgcn.v_sqrt_f32 outs(%arg0) ins(%arg1) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  // Consumer reads vgpr<1> and vgpr<3>, not the written vgpr<0>.
  amdgcn.v_add_f32 outs(%arg2) ins(%arg1, %arg3) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<3>)
  return
}
