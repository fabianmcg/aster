// RUN: aster-opt %s --verify-roundtrip

// Floating-point conversion roundtrip kernel.
// Computes per-lane:
//   f32  = sitofp a[i]  : i32 -> f32
//   f64  = extf   f32   : f32 -> f64
//   f32b = truncf f64   : f64 -> f32
//   r    = fptosi f32b  : f32 -> i32
// For inputs in [-1000, 1000) every i32 is exactly representable in f32,
// so the roundtrip is exact and r == a[i].
// Arguments (in buffer_arg order, matching compile_and_run input_data + output_data):
//   arg0: a   (read_only)  -- input i32 values
//   arg1: out (write_only) -- roundtrip i32 results

amdgcn.module @fpconv_mod target = #amdgcn.target<gfx950> {

  func.func private @load_two_ptrs()
      -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) {
    %a_ptr   = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %out_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %a_ptr, %out_ptr
      : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  }

  amdgcn.kernel @fpconv_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %a_ptr, %out_ptr = func.call @load_two_ptrs()
      : () -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)

    // Per-lane byte offset: threadid.x * 4
    %threadidx_x = amdgcn.thread_id x : !amdgcn.vgpr
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_alloc) ins(%c2, %threadidx_x)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c0 = arith.constant 0 : i32

    // Load a[tid].
    %a_alloc = amdgcn.alloca : !amdgcn.vgpr
    %a_val, %tok_a = amdgcn.global_load_dword dest %a_alloc addr %a_ptr
      offset d(%voffset) + c(%c0)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.read_token<flat>

    amdgcn.s_waitcnt vmcnt = 0

    // sitofp i32 -> f32
    %f32_dst = amdgcn.alloca : !amdgcn.vgpr
    %f32_val = lsir.sitofp f32 from i32 %f32_dst, %a_val
      : !amdgcn.vgpr, !amdgcn.vgpr

    // extf f32 -> f64 (f64 occupies two consecutive VGPRs)
    %f64_dst = lsir.alloca : !amdgcn.vgpr<[? + 2]>
    %f64_val = lsir.extf f64 from f32 %f64_dst, %f32_val
      : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr

    // truncf f64 -> f32
    %f32b_dst = amdgcn.alloca : !amdgcn.vgpr
    %f32b_val = lsir.truncf f32 from f64 %f32b_dst, %f64_val
      : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>

    // fptosi f32 -> i32
    %r_dst = amdgcn.alloca : !amdgcn.vgpr
    %r_val = lsir.fptosi i32 from f32 %r_dst, %f32b_val
      : !amdgcn.vgpr, !amdgcn.vgpr

    // Store result.
    %tok_out = amdgcn.global_store_dword data %r_val addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.write_token<flat>

    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
