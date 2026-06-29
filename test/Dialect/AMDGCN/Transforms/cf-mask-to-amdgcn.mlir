// RUN: aster-opt %s --aster-codegen --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: kernel @k_wave32_get_set
// CHECK:         %[[ELO:.*]] = alloca : !amdgcn.exec_lo<0>
// CHECK:         %[[S0:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:         %[[MASK_REG:.*]] = lsir.copy %[[S0]], %[[ELO]] : !amdgcn.sgpr, !amdgcn.exec_lo<0>
// CHECK:         %[[ELO2:.*]] = alloca : !amdgcn.exec_lo<0>
// CHECK:         lsir.copy %[[ELO2]], %[[MASK_REG]] : !amdgcn.exec_lo<0>, !amdgcn.sgpr
amdgcn.module @m target = <gfx1250> {
  amdgcn.kernel @k_wave32_get_set {
    %mask = aster_utils.get_cf_mask : i32
    aster_utils.set_cf_mask %mask : i32
    end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @k_get_set
// CHECK:         %[[ELO:.*]] = alloca : !amdgcn.exec_lo<0>
// CHECK:         %[[EHI:.*]] = alloca : !amdgcn.exec_hi<0>
// CHECK:         %[[EXEC:.*]] = make_register_range %[[ELO]], %[[EHI]] : !amdgcn.exec_lo<0>, !amdgcn.exec_hi<0>
// CHECK:         %[[SGPR:.*]] = lsir.alloca : !amdgcn.sgpr<[? + 2]>
// CHECK:         %[[MASK_REG:.*]] = lsir.copy %[[SGPR]], %[[EXEC]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.exec<0>
// CHECK:         lsir.copy {{.*}}, %[[MASK_REG]] : !amdgcn.exec<0>, !amdgcn.sgpr<[? + 2]>
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_get_set {
    %mask = aster_utils.get_cf_mask : i64
    aster_utils.set_cf_mask %mask : i64
    end_kernel
  }
}

// -----

// set_cf_mask with condition (wave64): EXEC = saved & cond via s_and_b64.
// CHECK-LABEL: kernel @k_wave64_set_with_cond
// CHECK:         s_and_b64 outs({{.*}}, {{.*}}) ins({{.*}}, {{.*}})
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_wave64_set_with_cond {
    %mask = aster_utils.get_cf_mask : i64
    %tid = aster_utils.thread_id x
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %tid, %c0 : i32
    aster_utils.set_cf_mask %mask, %cond : i64
    end_kernel
  }
}

// -----

// set_cf_mask with complement (wave64): EXEC = saved & ~cond via s_andn2_b64.
// CHECK-LABEL: kernel @k_wave64_set_with_complement
// CHECK:         s_andn2_b64 outs({{.*}}, {{.*}}) ins({{.*}}, {{.*}})
amdgcn.module @m target = <gfx942> {
  amdgcn.kernel @k_wave64_set_with_complement {
    %mask = aster_utils.get_cf_mask : i64
    %tid = aster_utils.thread_id x
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %tid, %c0 : i32
    aster_utils.set_cf_mask %mask, %cond{complement} : i64
    end_kernel
  }
}

// -----

// set_cf_mask with condition (wave32): EXEC = saved & cond via s_and_b32.
// CHECK-LABEL: kernel @k_wave32_set_with_cond
// CHECK:         s_and_b32 outs({{.*}}, {{.*}}) ins({{.*}}, {{.*}})
amdgcn.module @m target = <gfx1250> {
  amdgcn.kernel @k_wave32_set_with_cond {
    %mask = aster_utils.get_cf_mask : i32
    %tid = aster_utils.thread_id x
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %tid, %c0 : i32
    aster_utils.set_cf_mask %mask, %cond : i32
    end_kernel
  }
}

// -----

// set_cf_mask with complement (wave32): EXEC = saved & ~cond via s_andn2_b32.
// CHECK-LABEL: kernel @k_wave32_set_with_complement
// CHECK:         s_andn2_b32 outs({{.*}}, {{.*}}) ins({{.*}}, {{.*}})
amdgcn.module @m target = <gfx1250> {
  amdgcn.kernel @k_wave32_set_with_complement {
    %mask = aster_utils.get_cf_mask : i32
    %tid = aster_utils.thread_id x
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %tid, %c0 : i32
    aster_utils.set_cf_mask %mask, %cond{complement} : i32
    end_kernel
  }
}
