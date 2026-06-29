// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{debug-stalls=false})))" | FileCheck %s

!s = !amdgcn.sgpr

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // Two SCC cmpi+select chains using separate SCC allocas must NOT interleave.
  // Both allocas map to the single physical SCC register, so the second
  // comparison still clobbers SCC; the WAR/WAW edges added by the scheduler
  // ensure each cmpi/select pair completes before the next pair begins.
  // CHECK-LABEL: kernel @scc_serialize_cmpi_select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         lsir.cmpi
  // CHECK-NEXT:    lsir.select
  // CHECK:         end_kernel
  amdgcn.kernel @scc_serialize_cmpi_select {
    %s0 = amdgcn.alloca : !s
    %s1 = amdgcn.alloca : !s
    %s2 = amdgcn.alloca : !s
    %s3 = amdgcn.alloca : !s
    %scc_a = lsir.alloca : !amdgcn.scc
    %scc_b = lsir.alloca : !amdgcn.scc
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cmp_a = lsir.cmpi i32 eq %scc_a, %s0, %c0 : !amdgcn.scc, !s, i32
    lsir.select %s1, %cmp_a, %s0, %c1 : !s, !amdgcn.scc, !s, i32
    %cmp_b = lsir.cmpi i32 eq %scc_b, %s2, %c0 : !amdgcn.scc, !s, i32
    lsir.select %s3, %cmp_b, %s2, %c1 : !s, !amdgcn.scc, !s, i32
    amdgcn.end_kernel
  }

}
