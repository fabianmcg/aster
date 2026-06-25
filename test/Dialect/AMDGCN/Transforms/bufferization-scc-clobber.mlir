// RUN: aster-opt %s --aster-amdgcn-bufferization --split-input-file | FileCheck %s
// RUN: aster-opt %s --aster-amdgcn-bufferization --amdgcn-pre-coloring-legalization --split-input-file | FileCheck %s --check-prefix=LEGAL

// SCC clobbered by a non-value-semantic out (the s_add_u32 carry-out alloca).
// The compare's SCC must be promoted to SGPR before the clobber.

// CHECK-LABEL: kernel @scc_clobber_non_value_out {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc<0>
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[A:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[B:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[DST:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC]]) ins(%[[A]], %[[B]])
// CHECK:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// CHECK:         s_add_u32 outs(%[[DST]], %[[SCC_ALLOC]])
// CHECK:         test_inst ins %[[PROMOTED]]
// CHECK:         end_kernel

// LEGAL-LABEL: kernel @scc_clobber_non_value_out {
// LEGAL:         %[[SCC0:.*]] = s_cmp_eq_i32
// LEGAL:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// LEGAL:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// LEGAL:         s_add_u32
// LEGAL:         test_inst ins %[[PROMOTED]]
// LEGAL:         end_kernel
amdgcn.module @scc_clobber_non_value_out_mod target = <gfx942> {
  amdgcn.kernel @scc_clobber_non_value_out {
    %scc = alloca : !amdgcn.scc
    %a = alloca : !amdgcn.sgpr
    %b = alloca : !amdgcn.sgpr
    %dst = alloca : !amdgcn.sgpr
    %scc_alloc = alloca : !amdgcn.scc<0>
    %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    s_add_u32 outs(%dst, %scc_alloc) ins(%a, %b) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %scc0 : (!amdgcn.scc) -> ()
    end_kernel
  }
}

// -----

// SCC clobbered and later consumed via lsir.cond_br, requiring a restore copy
// (SGPR -> SCC). After legalization, the save becomes s_cselect_b32 and the
// restore becomes s_cmp_eq_u32.

// CHECK-LABEL: kernel @scc_clobber_with_restore {
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC]])
// CHECK:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// CHECK:         s_cmp_eq_i32 outs(%[[SCC]])
// CHECK:         %[[RESTORED:.*]] = lsir.copy %[[SCC]], %[[PROMOTED]]
// CHECK:         lsir.cond_br %[[RESTORED]]

// LEGAL-LABEL: kernel @scc_clobber_with_restore {
// LEGAL:         %[[SCC:.*]] = alloca : !amdgcn.scc
// LEGAL:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC]])
// LEGAL:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// LEGAL:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// LEGAL:         s_cmp_eq_i32 outs(%[[SCC]])
// LEGAL:         s_cmp_eq_u32 outs(%[[SCC]]) ins(%[[PROMOTED]]
// LEGAL:         lsir.cond_br %[[SCC]]
amdgcn.module @scc_clobber_with_restore_mod target = <gfx942> {
  amdgcn.kernel @scc_clobber_with_restore {
    %scc = alloca : !amdgcn.scc
    %a = alloca : !amdgcn.sgpr
    %b = alloca : !amdgcn.sgpr
    %c = alloca : !amdgcn.sgpr
    %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %scc1 : (!amdgcn.scc) -> ()
    lsir.cond_br %scc0 : !amdgcn.scc, ^bb1, ^bb2
  ^bb1:
    end_kernel
  ^bb2:
    end_kernel
  }
}

// -----

// No clobber: SCC defined and used immediately with no intervening clobbering
// op. No promotion should be inserted.

// CHECK-LABEL: kernel @no_clobber {
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC]])
// CHECK-NOT:     lsir.copy
// CHECK:         test_inst ins %[[SCC0]]
// CHECK:         end_kernel

// LEGAL-LABEL: kernel @no_clobber {
// LEGAL-NOT:     lsir.copy
// LEGAL-NOT:     s_cselect_b32
amdgcn.module @no_clobber_mod target = <gfx942> {
  amdgcn.kernel @no_clobber {
    %scc = alloca : !amdgcn.scc
    %a = alloca : !amdgcn.sgpr
    %b = alloca : !amdgcn.sgpr
    %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %scc0 : (!amdgcn.scc) -> ()
    end_kernel
  }
}
