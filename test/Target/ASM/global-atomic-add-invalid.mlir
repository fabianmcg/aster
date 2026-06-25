// RUN: aster-opt --split-input-file --verify-diagnostics %s

// Negative test: a value-semantic result on global_atomic_add_f32 without sc0
// must be rejected by the verifier.

func.func @atomic_add_f32_value_result_no_sc0(
    %addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr,
    %data: !amdgcn.vgpr, %dst0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %c0 = arith.constant 0 : i32
  // expected-error@+1 {{result may only be produced when sc0 is set}}
  %result, %tok = amdgcn.global_atomic_add_f32
      dst0 %dst0 data %data addr %addr_range offset c(%c0) :
      outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>)
      mods(i32) -> !amdgcn.write_token<flat>
  return %result : !amdgcn.vgpr
}

// -----

// Negative test: a value-semantic result on global_atomic_add without sc0
// must be rejected by the verifier.

func.func @atomic_add_i32_value_result_no_sc0(
    %addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr,
    %data: !amdgcn.vgpr, %dst0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %c0 = arith.constant 0 : i32
  // expected-error@+1 {{result may only be produced when sc0 is set}}
  %result, %tok = amdgcn.global_atomic_add
      dst0 %dst0 data %data addr %addr_range offset c(%c0) :
      outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>)
      mods(i32) -> !amdgcn.write_token<flat>
  return %result : !amdgcn.vgpr
}
