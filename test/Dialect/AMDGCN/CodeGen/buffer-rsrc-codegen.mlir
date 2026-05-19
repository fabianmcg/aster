// RUN: aster-opt %s --aster-codegen | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<
  !ptr.ptr<#amdgcn.addr_space<global, read_write>> = #ptr.spec<size = 64, abi = 64, preferred = 64>,
  !ptr.ptr<#amdgcn.addr_space<buffer, read_write>> = #ptr.spec<size = 128, abi = 128, preferred = 128>
>} {

// CHECK-LABEL:   func.func @test_make_buffer_rsrc(
// CHECK-SAME:      %[[BASE:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[NREC:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[STRIDE:.*]]: !amdgcn.sgpr) {
// CHECK:           amdgcn.make_buffer_rsrc %[[BASE]], %[[NREC]], {{.*}} cache_swizzle = false, swizzle_enable = false, flags = 131072
// CHECK:         }
  func.func @test_make_buffer_rsrc(
      %base: !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %num_records: i32,
      %stride: i32)
      attributes {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, !amdgcn.sgpr) -> ()} {
    %rsrc = amd_gpu.make_buffer_rsrc %base, %num_records, %stride, flags = 131072
            : (!ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32, i32)
            -> !ptr.ptr<#amdgcn.addr_space<buffer, read_write>>
    return {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, !amdgcn.sgpr) -> ()}
  }

// CHECK-LABEL:   func.func @test_buffer_load(
// CHECK-SAME:      %[[BASE:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[NREC:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[RSRC:.*]] = amdgcn.make_buffer_rsrc %[[BASE]], %[[NREC]], {{.*}} cache_swizzle = false, swizzle_enable = false
// CHECK:           %[[ALLOCA:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL:.*]], {{.*}} = amdgcn.buffer_load_dword dest %[[ALLOCA]] addr %[[RSRC]] offset u({{.*}}) + off_idx(%[[VOFF]]) + c({{.*}}) {offen}
// CHECK:           return %[[VAL]] : !amdgcn.vgpr
// CHECK:         }
  func.func @test_buffer_load(
      %base: !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %num_records: i32,
      %voffset: i32)
      -> i32
      attributes {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
    %c0_stride = arith.constant 0 : i32
    %rsrc = amd_gpu.make_buffer_rsrc %base, %num_records, %c0_stride, flags = 131072
            : (!ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32, i32)
            -> !ptr.ptr<#amdgcn.addr_space<buffer, read_write>>
    %addr = ptr.ptr_add %rsrc, %voffset
            : !ptr.ptr<#amdgcn.addr_space<buffer, read_write>>, i32
    %val = ptr.load %addr : !ptr.ptr<#amdgcn.addr_space<buffer, read_write>> -> i32
    return {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} %val : i32
  }

// CHECK-LABEL:   func.func @test_buffer_store(
// CHECK-SAME:      %[[BASE:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[NREC:.*]]: !amdgcn.sgpr,
// CHECK-SAME:      %[[DATA:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[VOFF:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[RSRC:.*]] = amdgcn.make_buffer_rsrc
// CHECK:           amdgcn.buffer_store_dword data %[[DATA]] addr %[[RSRC]] offset u({{.*}}) + off_idx(%[[VOFF]]) + c({{.*}}) {offen}
// CHECK:         }
  func.func @test_buffer_store(
      %base: !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %num_records: i32,
      %data: i32,
      %voffset: i32)
      attributes {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()} {
    %c0_stride = arith.constant 0 : i32
    %rsrc = amd_gpu.make_buffer_rsrc %base, %num_records, %c0_stride, flags = 131072
            : (!ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32, i32)
            -> !ptr.ptr<#amdgcn.addr_space<buffer, read_write>>
    %addr = ptr.ptr_add %rsrc, %voffset
            : !ptr.ptr<#amdgcn.addr_space<buffer, read_write>>, i32
    ptr.store %data, %addr : i32, !ptr.ptr<#amdgcn.addr_space<buffer, read_write>>
    return {abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()}
  }
}
