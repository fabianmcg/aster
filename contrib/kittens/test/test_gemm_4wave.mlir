// 4-wave GEMM kernel: C[32x32] = A[32xK] @ B[32xK]^T
//
// 2x2 wave grid (waves_m=2, waves_n=2):
//   Wave 0: C[0:16,  0:16]  = A[0:16,  :K] @ B[0:16,  :K]^T
//   Wave 1: C[0:16,  16:32] = A[0:16,  :K] @ B[16:32, :K]^T
//   Wave 2: C[16:32, 0:16]  = A[16:32, :K] @ B[0:16,  :K]^T
//   Wave 3: C[16:32, 16:32] = A[16:32, :K] @ B[16:32, :K]^T
//
// Each wave independently loads its A and B tiles based on its position
// in the 2x2 grid. Uses direct global loads (no LDS).
//
// Template parameters:
//   {{K}}         - K dimension (must be divisible by 16)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4

amdgcn.module @kittens_gemm_4wave target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From indexing.mlir
  func.func private @wave_id() -> index

  // From kittens/tiles_16x16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !rt_A_f16
  func.func private @load_B_f16(!sx2, index, index, index) -> !rt_B_f16
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index)

  // 4-wave GEMM kernel (256 threads = 4 waves)
  // Input:  A [32xK f16, row-major], B [32xK f16, row-major]
  // Output: C [32x32 f32, row-major]
  amdgcn.kernel @gemm_4wave arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 128 : index             // 32 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position in 2x2 grid:
    //   wave_m = wave_id floordiv 2 (row in wave grid)
    //   wave_n = wave_id mod 2     (col in wave grid)
    //   m_offset = wave_m * 16     (A row offset, C row offset)
    //   n_offset = wave_n * 16     (B row offset, C col offset)
    %wid = func.call @wave_id() : () -> index
    %m_offset = affine.apply affine_map<()[wid] -> ((wid floordiv 2) * 16)>()[%wid]
    %n_offset = affine.apply affine_map<()[wid] -> ((wid mod 2) * 16)>()[%wid]

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: each wave iterates over K tiles independently
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // Load A tile at this wave's row offset
      %A_tile = func.call @load_A_f16(%A_ptr, %m_offset, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !rt_A_f16

      // Load B tile at this wave's column offset (B row = n_offset)
      %B_tile = func.call @load_B_f16(%B_ptr, %n_offset, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !rt_B_f16

      // MFMA: acc += A_tile @ B_tile^T
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result at this wave's (m, n) offset in C
    func.call @store_C_f32(%C_final, %C_ptr, %m_offset, %n_offset, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
