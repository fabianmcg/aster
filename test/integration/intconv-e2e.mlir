// RUN: aster-opt %s --verify-roundtrip

// Integer conversion kernel.
// Computes per-lane:
//   roundtrip = extui(trunci(A[i] : i32 -> i16) : i16 -> i32)  = A[i] & 0xFFFF
//   select    = A[i] < B[i] ? A[i] : B[i]                      = min(A[i], B[i])
// Arguments (in buffer_arg order, matching compile_and_run input_data + output_data):
//   arg0: A (read_only)          -- first input
//   arg1: B (read_only)          -- second input
//   arg2: roundtrip (write_only) -- trunci/extui result
//   arg3: select_out (write_only)-- select result

amdgcn.module @intconv_mod target = #amdgcn.target<gfx950> {

  func.func private @load_four_ptrs()
      -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
          !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) {
    %a_ptr        = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %b_ptr        = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %roundtrip_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr<[? + 2]>
    %select_ptr   = amdgcn.load_arg 3 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %a_ptr, %b_ptr, %roundtrip_ptr, %select_ptr
      : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
        !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  }

  amdgcn.kernel @intconv_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %a_ptr, %b_ptr, %roundtrip_ptr, %select_ptr = func.call @load_four_ptrs()
      : () -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
               !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)

    // Per-lane byte offset: threadid.x * 4
    %threadidx_x = amdgcn.thread_id x : !amdgcn.vgpr
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_alloc) ins(%c2, %threadidx_x)
      : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %c0 = arith.constant 0 : i32

    // Load A[tid] and B[tid].
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

    // Roundtrip: trunci i32->i16 then extui i16->i32 (zero-extends low 16 bits).
    %trunc_dst = amdgcn.alloca : !amdgcn.vgpr
    %truncated = lsir.trunci i16 from i32 %trunc_dst, %a_val
      : !amdgcn.vgpr, !amdgcn.vgpr

    %ext_dst = amdgcn.alloca : !amdgcn.vgpr
    %extended = lsir.extui i32 from i16 %ext_dst, %truncated
      : !amdgcn.vgpr, !amdgcn.vgpr

    // Select: VCC-based select; pick a_val when a_val < b_val (slt), else b_val.
    %vcc_a = lsir.alloca : !amdgcn.vcc
    %cmp = lsir.cmpi i32 slt %vcc_a, %a_val, %b_val
      : !amdgcn.vcc, !amdgcn.vgpr, !amdgcn.vgpr

    %sel_dst = amdgcn.alloca : !amdgcn.vgpr
    %selected = lsir.select %sel_dst, %cmp, %a_val, %b_val
      : !amdgcn.vgpr, !amdgcn.vcc, !amdgcn.vgpr, !amdgcn.vgpr

    // Store roundtrip and select results.
    %tok_rt = amdgcn.global_store_dword data %extended addr %roundtrip_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.write_token<flat>

    %tok_sel = amdgcn.global_store_dword data %selected addr %select_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr)
        mods(i32) -> !amdgcn.write_token<flat>

    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
