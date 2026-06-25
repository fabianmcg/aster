// RUN: aster-opt %s --verify-roundtrip
//
// E2E test for divergent control-flow lowering via save_cf_mask / restore_cf_mask.
//
// A divergent condition (tid_x < 32) gates a per-lane global store. Lanes
// 0..31 satisfy the condition and write 1; lanes 32..63 skip the body and
// leave the output at 0 (pre-initialized by the host).
//
// The test starts from scf.if with a divergent per-lane condition and flows
// through aster-convert-scf-control-flow (inserts save_cf_mask / restore_cf_mask)
// and then aster-codegen (lowers to s_and_saveexec_b64 / s_or_b64).

amdgcn.module @divergent_cf_mod target = #amdgcn.target<gfx942> {

  // block=(64,1,1), grid=(1,1,1).
  // output: 64 dwords pre-filled with 0.
  // Expected after kernel: output[0..31] = 1, output[32..63] = 0.
  amdgcn.kernel @divergent_cf_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %tid_vgpr = amdgcn.thread_id x : !amdgcn.vgpr

    // Per-lane byte offset: tid_x * 4.
    %c2  = arith.constant 2  : i32
    %c0  = arith.constant 0  : i32
    %c1v = arith.constant 1  : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid_vgpr)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Materialize the write value (1) into a VGPR.
    %v1_a = amdgcn.alloca : !amdgcn.vgpr
    %v1   = amdgcn.v_mov_b32 outs(%v1_a) ins(%c1v)
      : outs(!amdgcn.vgpr) ins(i32)

    // Use aster_utils.thread_id (i32) to form the divergent condition for
    // scf.if. The comparison is per-lane (different result per thread).
    %c32 = arith.constant 32 : i32
    %tid_i32 = aster_utils.thread_id x
    %cond = arith.cmpi slt, %tid_i32, %c32 : i32

    // scf.if with a divergent condition: aster-convert-scf-control-flow will
    // insert save_cf_mask (-> s_and_saveexec_b64) and restore_cf_mask
    // (-> s_or_b64) around this region.
    scf.if %cond {
      %tok = amdgcn.global_store_dword data %v1 addr %out_ptr
        offset d(%voffset) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    }
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
