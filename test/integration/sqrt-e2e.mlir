// RUN: aster-opt %s --verify-roundtrip
//
// E2E test: per-lane f32 sqrt via lsir.sqrtf → v_sqrt_f32.
//
// Each lane loads its f32 input, computes sqrt, and stores the result.
// Correctness verified against numpy.sqrt in test_sqrt_e2e.py.
//
// Arguments:
//   arg0: input f32 buffer  (read_write, used as InOutArray)
//   arg1: output f32 buffer (read_write)

amdgcn.module @sqrt_e2e_mod target = #amdgcn.target<gfx942> {
  func.func @sqrt_f32_kernel(
      %in_ptr  : !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
      %out_ptr : !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {

    %c4  = arith.constant 4 : i32
    %c0  = arith.constant 0 : i32

    %tid  = gpu.thread_id x
    %bdim = gpu.block_dim x
    %bid  = gpu.block_id x

    // flat element index for this thread
    %flat_idx = affine.apply affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
                    (%bid, %tid)[%bdim]
    %byte_off_i = affine.apply affine_map<(d0) -> (d0 * 4)>(%flat_idx)
    %byte_off   = arith.index_cast %byte_off_i : index to i32

    %in_elem_ptr = ptr.ptr_add %in_ptr, %byte_off
                    : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    %val = ptr.load %in_elem_ptr
             : !ptr.ptr<#amdgcn.addr_space<global, read_write>> -> f32

    // IEEE sqrt via lsir.sqrtf → v_sqrt_f32
    %dst_v = lsir.alloca : !amdgcn.vgpr
    %val_v = lsir.to_reg %val : f32 -> !amdgcn.vgpr
    %res_v = lsir.sqrtf f32 %dst_v, %val_v : !amdgcn.vgpr, !amdgcn.vgpr
    %res   = lsir.from_reg %res_v : !amdgcn.vgpr -> f32

    %out_elem_ptr = ptr.ptr_add %out_ptr, %byte_off
                     : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    ptr.store %res, %out_elem_ptr
        : f32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>

    func.return
  }
}
