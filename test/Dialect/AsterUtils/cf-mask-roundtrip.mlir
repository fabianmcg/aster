// RUN: aster-opt %s | aster-opt | FileCheck %s

// CHECK-LABEL: func.func @test_get_cf_mask_i64
func.func @test_get_cf_mask_i64() -> i64 {
  // CHECK: %[[MASK:.*]] = aster_utils.get_cf_mask : i64
  %mask = aster_utils.get_cf_mask : i64
  return %mask : i64
}

// CHECK-LABEL: func.func @test_get_cf_mask_i32
func.func @test_get_cf_mask_i32() -> i32 {
  // CHECK: %[[MASK:.*]] = aster_utils.get_cf_mask : i32
  %mask = aster_utils.get_cf_mask : i32
  return %mask : i32
}

// CHECK-LABEL: func.func @test_set_cf_mask_i64
func.func @test_set_cf_mask_i64() {
  // CHECK: %[[M:.*]] = arith.constant
  %m = arith.constant 0 : i64
  // CHECK: aster_utils.set_cf_mask %[[M]] : i64
  aster_utils.set_cf_mask %m : i64
  return
}

// CHECK-LABEL: func.func @test_set_cf_mask_i32
func.func @test_set_cf_mask_i32() {
  // CHECK: %[[M:.*]] = arith.constant
  %m = arith.constant 0 : i32
  // CHECK: aster_utils.set_cf_mask %[[M]] : i32
  aster_utils.set_cf_mask %m : i32
  return
}

// CHECK-LABEL: func.func @test_set_cf_mask_with_cond_i64
func.func @test_set_cf_mask_with_cond_i64(%cond: i1) {
  // CHECK: %[[M:.*]] = arith.constant
  %m = arith.constant 0 : i64
  // CHECK: aster_utils.set_cf_mask %[[M]], %{{.*}} : i64
  aster_utils.set_cf_mask %m, %cond : i64
  return
}

// CHECK-LABEL: func.func @test_set_cf_mask_with_complement_i64
func.func @test_set_cf_mask_with_complement_i64(%cond: i1) {
  // CHECK: %[[M:.*]] = arith.constant
  %m = arith.constant 0 : i64
  // CHECK: aster_utils.set_cf_mask %[[M]], %{{.*}}{complement} : i64
  aster_utils.set_cf_mask %m, %cond {complement} : i64
  return
}
