// RUN: aster-opt %s --verify-roundtrip

// Integer unsigned divide and modulo kernel.
// Computes div = A[i] / B[i] and mod = A[i] % B[i] for each lane, then
// stores the results to separate output buffers.
// Arguments (in buffer_arg order, matching compile_and_run input_data + output_data):
//   arg0: A (read_only)  -- dividend
//   arg1: B (read_only)  -- divisor
//   arg2: DIV (write_only) -- quotient output
//   arg3: MOD (write_only) -- remainder output

amdgcn.module @intdivmod_mod target = #amdgcn.target<gfx950> {

  func.func private @load_four_ptrs()
      -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
          !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) {
    %a_ptr   = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %b_ptr   = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %div_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr<[? + 2]>
    %mod_ptr = amdgcn.load_arg 3 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %a_ptr, %b_ptr, %div_ptr, %mod_ptr
      : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
        !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  }

  amdgcn.kernel @intdivmod_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %a_ptr, %b_ptr, %div_ptr, %mod_ptr = func.call @load_four_ptrs()
      : () -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
               !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)

    // Per-lane byte offset: threadid.x * 4
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_alloc) ins(%c2, %threadidx_x)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr<0>)
    %c0 = arith.constant 0 : i32

    // Load A[tid] and B[tid]
    %a_alloc = amdgcn.alloca : !amdgcn.vgpr
    %a_val, %tok_a = amdgcn.global_load_dword dest %a_alloc addr %a_ptr
      offset d(%voffset) + c(%c0)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.read_token<flat>

    %b_alloc = amdgcn.alloca : !amdgcn.vgpr
    %b_val, %tok_b = amdgcn.global_load_dword dest %b_alloc addr %b_ptr
      offset d(%voffset) + c(%c0)
      : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.read_token<flat>

    amdgcn.s_waitcnt vmcnt = 0

    // Unsigned divide and modulo via lsir ops (lowered by -aster-to-amdgcn).
    %divdst = amdgcn.alloca : !amdgcn.vgpr
    %divq = lsir.divui i32 %divdst, %a_val, %b_val
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    %moddst = amdgcn.alloca : !amdgcn.vgpr
    %modr = lsir.remui i32 %moddst, %a_val, %b_val
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    // Store quotient and remainder
    %tok_div = amdgcn.global_store_dword data %divq addr %div_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.write_token<flat>

    %tok_mod = amdgcn.global_store_dword data %modr addr %mod_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.write_token<flat>

    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
