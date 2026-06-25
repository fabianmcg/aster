// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Verify ASM emission for global_atomic_add_f32 and global_atomic_add
// on CDNA3 (gfx942) and CDNA4 (gfx950) targets.

// CHECK-LABEL: Module: atomic_add_mod_cdna3
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK-LABEL: atomic_store_f32:
// CHECK:    global_atomic_add_f32 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off
// CHECK:    s_endpgm
// CHECK-LABEL: atomic_store_i32:
// CHECK:    global_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off
// CHECK:    s_endpgm

amdgcn.module @atomic_add_mod_cdna3 target = #amdgcn.target<gfx942> {
  amdgcn.kernel @atomic_store_f32 {
    %c0 = arith.constant 0 : i32
    %dst = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr_lo = amdgcn.alloca : !amdgcn.vgpr<4>
    %addr_hi = amdgcn.alloca : !amdgcn.vgpr<5>
    %addr = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
    %tok = amdgcn.global_atomic_add_f32
        dst0 %dst data %data addr %addr offset c(%c0) :
        outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>, !amdgcn.vgpr<[4 : 6]>)
        mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
  amdgcn.kernel @atomic_store_i32 {
    %c0 = arith.constant 0 : i32
    %dst = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr_lo = amdgcn.alloca : !amdgcn.vgpr<4>
    %addr_hi = amdgcn.alloca : !amdgcn.vgpr<5>
    %addr = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
    %tok = amdgcn.global_atomic_add
        dst0 %dst data %data addr %addr offset c(%c0) :
        outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>, !amdgcn.vgpr<[4 : 6]>)
        mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// CHECK-LABEL: Module: atomic_add_sc0_mod_cdna3
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK-LABEL: atomic_add_f32_sc0:
// CHECK:    global_atomic_add_f32 v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off sc0
// CHECK:    s_endpgm
// CHECK-LABEL: atomic_add_i32_sc0:
// CHECK:    global_atomic_add v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off sc0
// CHECK:    s_endpgm

amdgcn.module @atomic_add_sc0_mod_cdna3 target = #amdgcn.target<gfx942> {
  amdgcn.kernel @atomic_add_f32_sc0 {
    %c0 = arith.constant 0 : i32
    %result_reg = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr_lo = amdgcn.alloca : !amdgcn.vgpr<4>
    %addr_hi = amdgcn.alloca : !amdgcn.vgpr<5>
    %addr = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
    %tok = amdgcn.global_atomic_add_f32
        dst0 %result_reg data %data addr %addr offset c(%c0) {sc0} :
        outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>, !amdgcn.vgpr<[4 : 6]>)
        mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
  amdgcn.kernel @atomic_add_i32_sc0 {
    %c0 = arith.constant 0 : i32
    %result_reg = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr_lo = amdgcn.alloca : !amdgcn.vgpr<4>
    %addr_hi = amdgcn.alloca : !amdgcn.vgpr<5>
    %addr = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
    %tok = amdgcn.global_atomic_add
        dst0 %result_reg data %data addr %addr offset c(%c0) {sc0} :
        outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>, !amdgcn.vgpr<[4 : 6]>)
        mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// CHECK-LABEL: Module: atomic_add_mod_cdna4
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
// CHECK-LABEL: atomic_store_f32_cdna4:
// CHECK:    global_atomic_add_f32 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off
// CHECK:    s_endpgm

amdgcn.module @atomic_add_mod_cdna4 target = #amdgcn.target<gfx950> {
  amdgcn.kernel @atomic_store_f32_cdna4 {
    %c0 = arith.constant 0 : i32
    %dst = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr_lo = amdgcn.alloca : !amdgcn.vgpr<4>
    %addr_hi = amdgcn.alloca : !amdgcn.vgpr<5>
    %addr = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
    %tok = amdgcn.global_atomic_add_f32
        dst0 %dst data %data addr %addr offset c(%c0) :
        outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>, !amdgcn.vgpr<[4 : 6]>)
        mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}

// Verify with-return i32 atomic add with sc0 and sc1 (device scope) for CDNA4.
// CHECK-LABEL: Module: atomic_add_i32_ret_sc1_cdna4
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
// CHECK-LABEL: atomic_add_i32_sc0_sc1:
// CHECK:    global_atomic_add v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off sc0 sc1
// CHECK:    s_endpgm

amdgcn.module @atomic_add_i32_ret_sc1_cdna4 target = #amdgcn.target<gfx950> {
  amdgcn.kernel @atomic_add_i32_sc0_sc1 {
    %c0 = arith.constant 0 : i32
    %result_reg = amdgcn.alloca : !amdgcn.vgpr<0>
    %data = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr_lo = amdgcn.alloca : !amdgcn.vgpr<4>
    %addr_hi = amdgcn.alloca : !amdgcn.vgpr<5>
    %addr = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
    %tok = amdgcn.global_atomic_add
        dst0 %result_reg data %data addr %addr offset c(%c0) {sc0, sc1} :
        outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>, !amdgcn.vgpr<[4 : 6]>)
        mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
