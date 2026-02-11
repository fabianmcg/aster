// 2-wave GEMM kernel with LDS: C[32x16] = A[32xK] @ B[16xK]^T
//
// 2x1 wave grid (waves_m=2, waves_n=1):
//   Wave 0: C[0:16,  0:16] = A[0:16,  :K] @ B[0:16, :K]^T
//   Wave 1: C[16:32, 0:16] = A[16:32, :K] @ B[0:16, :K]^T
//
// LDS layout: 3 tiles (A0, A1, B_shared)
//   - Each wave loads its own A tile into its LDS buffer
//   - Both waves redundantly load the shared B tile (same data, same address)
//   - s_barrier synchronizes all 128 threads before LDS reads
//
// Template parameters:
//   {{K}}         - K dimension (must be divisible by 16)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2
//   {{LDS_MODE}}  - LDS addressing mode: 0=nopad, 1=padded, 2=xor_swizzle
//   {{LDS_BYTES}} - Total LDS bytes: 1536 (nopad/swizzle) or 1632 (padded)

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4

amdgcn.module @kittens_gemm_2wave_lds target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From indexing.mlir
  func.func private @wave_id() -> index

  // From kittens/tiles_16x16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index)

  // From kittens/lds_16x16.mlir - 2-wave allocation (mode-specific)
  func.func private @alloc_lds_2wave_1buffer_nopad() -> (index, index, index)
  func.func private @alloc_lds_2wave_1buffer() -> (index, index, index)
  func.func private @alloc_lds_2wave_1buffer_xor_swizzle() -> (index, index, index)
  func.func private @lds_barrier()

  // From kittens/lds_transfers.mlir - transfers (mode-specific)
  // Mode 0: non-padded
  func.func private @load_global_to_lds_nopad_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_nopad_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_nopad_f16(index) -> !rt_B_f16
  // Mode 1: padded
  func.func private @load_global_to_lds_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_f16(index) -> !rt_B_f16
  // Mode 2: XOR swizzle
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_xor_swizzle_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_xor_swizzle_f16(index) -> !rt_B_f16

  // Helper: allocate LDS for 2-wave based on mode
  func.func private @alloc_lds_2wave(%mode: index) -> (index, index, index) {
    %lds_A0, %lds_A1, %lds_B = scf.index_switch %mode -> index, index, index
    case 0 {
      %a0, %a1, %b = func.call @alloc_lds_2wave_1buffer_nopad() : () -> (index, index, index)
      scf.yield %a0, %a1, %b : index, index, index
    }
    case 1 {
      %a0, %a1, %b = func.call @alloc_lds_2wave_1buffer() : () -> (index, index, index)
      scf.yield %a0, %a1, %b : index, index, index
    }
    default {
      %a0, %a1, %b = func.call @alloc_lds_2wave_1buffer_xor_swizzle() : () -> (index, index, index)
      scf.yield %a0, %a1, %b : index, index, index
    }
    return %lds_A0, %lds_A1, %lds_B : index, index, index
  }

  // Helper: cooperative global -> LDS load based on mode
  func.func private @global_to_lds(
      %mode: index, %lds_base: index, %ptr: !sx2,
      %m: index, %n: index, %stride: index
  ) {
    scf.index_switch %mode
    case 0 {
      func.call @load_global_to_lds_nopad_f16(%lds_base, %ptr, %m, %n, %stride)
          : (index, !sx2, index, index, index) -> ()
      scf.yield
    }
    case 1 {
      func.call @load_global_to_lds_f16(%lds_base, %ptr, %m, %n, %stride)
          : (index, !sx2, index, index, index) -> ()
      scf.yield
    }
    default {
      func.call @load_global_to_lds_xor_swizzle_f16(%lds_base, %ptr, %m, %n, %stride)
          : (index, !sx2, index, index, index) -> ()
      scf.yield
    }
    return
  }

  // Helper: LDS -> register tile load based on mode
  func.func private @lds_to_reg_A(%mode: index, %lds_base: index) -> !rt_A_f16 {
    %tile = scf.index_switch %mode -> !rt_A_f16
    case 0 {
      %t = func.call @load_lds_to_register_A_nopad_f16(%lds_base) : (index) -> !rt_A_f16
      scf.yield %t : !rt_A_f16
    }
    case 1 {
      %t = func.call @load_lds_to_register_A_f16(%lds_base) : (index) -> !rt_A_f16
      scf.yield %t : !rt_A_f16
    }
    default {
      %t = func.call @load_lds_to_register_A_xor_swizzle_f16(%lds_base) : (index) -> !rt_A_f16
      scf.yield %t : !rt_A_f16
    }
    return %tile : !rt_A_f16
  }

  func.func private @lds_to_reg_B(%mode: index, %lds_base: index) -> !rt_B_f16 {
    %tile = scf.index_switch %mode -> !rt_B_f16
    case 0 {
      %t = func.call @load_lds_to_register_B_nopad_f16(%lds_base) : (index) -> !rt_B_f16
      scf.yield %t : !rt_B_f16
    }
    case 1 {
      %t = func.call @load_lds_to_register_B_f16(%lds_base) : (index) -> !rt_B_f16
      scf.yield %t : !rt_B_f16
    }
    default {
      %t = func.call @load_lds_to_register_B_xor_swizzle_f16(%lds_base) : (index) -> !rt_B_f16
      scf.yield %t : !rt_B_f16
    }
    return %tile : !rt_B_f16
  }

  // 2-wave GEMM kernel (128 threads = 2 waves) with LDS
  // Input:  A [32xK f16, row-major], B [16xK f16, row-major]
  // Output: C [32x16 f32, row-major]
  amdgcn.kernel @gemm_2wave_lds arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = {{LDS_BYTES}} : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mode = arith.constant {{LDS_MODE}} : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position: m_offset = wave_id * 16
    %wid = func.call @wave_id() : () -> index
    %m_offset = affine.apply affine_map<()[wid] -> (wid * 16)>()[%wid]

    // Allocate LDS: 3 tiles (A0 for wave 0, A1 for wave 1, B shared)
    %lds_A0, %lds_A1, %lds_B = func.call @alloc_lds_2wave(%mode)
        : (index) -> (index, index, index)

    // Select this wave's A LDS buffer: lds_A = lds_A0 + wave_id * (lds_A1 - lds_A0)
    // wave 0 -> lds_A0, wave 1 -> lds_A0 + (lds_A1 - lds_A0) = lds_A1
    %lds_A_stride = affine.apply affine_map<()[a0, a1] -> (a1 - a0)>()[%lds_A0, %lds_A1]
    %lds_A = affine.apply affine_map<()[base, wid, stride] -> (base + wid * stride)>()[%lds_A0, %wid, %lds_A_stride]

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: each wave iterates over K tiles
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // === Step 1: Cooperative load Global -> LDS ===
      // Each wave loads its own A tile (wave 0 -> A0, wave 1 -> A1)
      func.call @global_to_lds(%mode, %lds_A, %A_ptr, %m_offset, %k_offset, %stride_AB)
          : (index, index, !sx2, index, index, index) -> ()

      // Both waves redundantly load shared B tile (same data, same LDS address)
      func.call @global_to_lds(%mode, %lds_B, %B_ptr, %c0, %k_offset, %stride_AB)
          : (index, index, !sx2, index, index, index) -> ()

      // === Step 2: Cross-wave barrier ===
      func.call @lds_barrier() : () -> ()

      // === Step 3: Load LDS -> Register ===
      %A_tile = func.call @lds_to_reg_A(%mode, %lds_A) : (index, index) -> !rt_A_f16
      %B_tile = func.call @lds_to_reg_B(%mode, %lds_B) : (index, index) -> !rt_B_f16

      // === Step 4: Compute ===
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result at this wave's row offset in C
    func.call @store_C_f32(%C_final, %C_ptr, %m_offset, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
