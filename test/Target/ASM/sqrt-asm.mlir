// RUN: aster-opt %s \
// RUN:   --amdgcn-reg-alloc --amdgcn-late-waits --symbol-dce \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Check that v_sqrt_f32 and v_rsq_f32 produce correct assembly mnemonics.

// CHECK-LABEL: test_sqrt_asm:
// CHECK:       v_sqrt_f32 v{{[0-9]+}}, v{{[0-9]+}}
// CHECK:       s_endpgm

// CHECK-LABEL: test_rsq_asm:
// CHECK:       v_rsq_f32 v{{[0-9]+}}, v{{[0-9]+}}
// CHECK:       s_endpgm

amdgcn.module @sqrt_asm_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @test_sqrt_asm attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %src = amdgcn.alloca : !amdgcn.vgpr<0>
    %dst = amdgcn.alloca : !amdgcn.vgpr<1>
    amdgcn.v_sqrt_f32 outs(%dst) ins(%src)
        : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<0>)
    amdgcn.end_kernel
  }

  amdgcn.kernel @test_rsq_asm attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %src = amdgcn.alloca : !amdgcn.vgpr<0>
    %dst = amdgcn.alloca : !amdgcn.vgpr<1>
    amdgcn.v_rsq_f32 outs(%dst) ins(%src)
        : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<0>)
    amdgcn.end_kernel
  }
}
