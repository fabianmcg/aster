// RUN: aster-opt %s --aster-convert-scf-control-flow --aster-codegen | FileCheck %s
//
// Codegen lowering of aster_utils.save_cf_mask, aster_utils.set_cf_mask and
// aster_utils.restore_cf_mask after SCF-to-CF conversion:
//   Uniform condition  -> s_mov_b64 snapshot + s_mov_b64 restore; no saveexec,
//                         no s_and_b64, no s_or_b64 (set_cf_mask is erased).
//   Divergent condition -> s_mov_b64 on save; s_and_b64 on then entry; s_or_b64 on restore.
//   Divergent with else -> s_mov_b64 on save; s_and_b64 on then entry;
//                          s_andn2_b64 on else entry; s_or_b64 on restore.

amdgcn.module @cf_mask_codegen target = <gfx942> {

// CHECK-LABEL: kernel @test_uniform_cf_mask
// Uniform path: EXEC is snapshotted via s_mov_b64 and restored via s_mov_b64;
// no s_and_saveexec_b64, no s_and_b64, and no s_or_b64 are emitted.
// CHECK-NOT: s_and_saveexec_b64
// CHECK: s_mov_b64
// CHECK: s_mov_b64
// CHECK-NOT: s_and_b64
// CHECK-NOT: s_or_b64
// CHECK-NOT: save_cf_mask
// CHECK-NOT: restore_cf_mask
// CHECK-NOT: set_cf_mask
kernel @test_uniform_cf_mask {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // assume_uniform forces the cmpi result to be treated as scalar (SCC).
  %cond_raw = arith.cmpi slt, %c0, %c1 : i32
  %cond = aster_utils.assume_uniform %cond_raw : i1
  scf.if %cond {
  }
  end_kernel
}

// CHECK-LABEL: kernel @test_divergent_cf_mask
// Divergent path: s_mov_b64 on save, s_and_b64 on then entry, s_or_b64 on restore.
// CHECK: s_mov_b64
// CHECK: s_and_b64
// CHECK: s_or_b64
// CHECK-NOT: s_and_saveexec_b64
// CHECK-NOT: save_cf_mask
// CHECK-NOT: restore_cf_mask
// CHECK-NOT: set_cf_mask
kernel @test_divergent_cf_mask {
  // thread_id returns a divergent i32 value (one per lane).
  %tid = aster_utils.thread_id x
  %c0 = arith.constant 0 : i32
  // cmpi on a divergent value produces a divergent i1 condition (VCC).
  %cond = arith.cmpi sgt, %tid, %c0 : i32
  scf.if %cond {
  }
  end_kernel
}

// CHECK-LABEL: kernel @test_divergent_cf_mask_else
// Divergent path with else: s_mov_b64 on save, s_and_b64 on then entry,
// s_andn2_b64 on else entry (select else-lanes), and s_or_b64 on restore.
// CHECK: s_mov_b64
// CHECK: s_and_b64
// CHECK: s_andn2_b64
// CHECK: s_or_b64
// CHECK-NOT: s_and_saveexec_b64
// CHECK-NOT: save_cf_mask
// CHECK-NOT: restore_cf_mask
// CHECK-NOT: set_cf_mask
kernel @test_divergent_cf_mask_else {
  %tid = aster_utils.thread_id x
  %c0 = arith.constant 0 : i32
  %cond = arith.cmpi sgt, %tid, %c0 : i32
  scf.if %cond {
  } else {
  }
  end_kernel
}

} // amdgcn.module
