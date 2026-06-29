// RUN: aster-opt %s --verify-roundtrip
//
// E2E test for divergent control-flow lowering via save_cf_mask / restore_cf_mask.
//
// A divergent condition (tid_x < 32) gates a per-lane global store. Lanes
// 0..31 satisfy the condition and write 1; lanes 32..63 skip the body and
// leave the output at 0 (pre-initialized by the host).

// block=(64,1,1), grid=(1,1,1).
// output: 64 dwords pre-filled with 0.
// Expected after kernel: output[0..31] = 1, output[32..63] = 0.
amdgcn.module @divergent_cf_mod target = <gfx942> {
  func.func @divergent_cf_kernel(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %c1   = arith.constant 1 : i32
    %c4   = arith.constant 4 : i32
    %c32  = arith.constant 32 : i32
    %off  = arith.muli %tid, %c4 : i32
    %ptr  = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    %cond = arith.cmpi slt, %tid, %c32 : i32
    scf.if %cond {
      ptr.store %c1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }
    return
  }
}
