// RUN: aster-opt %s --verify-roundtrip --split-input-file
//
// E2E tests for scf.if lowering via aster-convert-scf-control-flow.
//
// Six kernels, each in its own module:
//   test_if_else_uniform            -- uniform condition; else branch; no results.
//   test_if_divergent_no_else       -- divergent condition; no else; no results.
//   test_if_divergent_with_else     -- divergent condition; else branch; no results.
//   test_if_divergent_with_results  -- divergent condition; stores tid or 0.
//   test_if_complex_condition_and   -- divergent AND condition; no else.
//   test_if_complex_condition_or    -- divergent OR condition; no else.

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = 1, output[32..63] = 2.
// Uniform threshold: both branches execute (no divergence).
amdgcn.module @scf_if_else_uniform_mod target = <gfx942> {
  func.func @test_if_else_uniform(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid   = aster_utils.thread_id x
    %c1    = arith.constant 1 : i32
    %c2    = arith.constant 2 : i32
    %c4    = arith.constant 4 : i32
    %c32   = arith.constant 32 : i32
    %off   = arith.muli %tid, %c4 : i32
    %ptr   = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    %c32_u = aster_utils.assume_uniform %c32 : i32
    %cond  = arith.cmpi slt, %tid, %c32_u : i32
    scf.if %cond {
      ptr.store %c1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    } else {
      ptr.store %c2, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = 1, output[32..63] = 0.
// Divergent condition, no else, no results.
amdgcn.module @scf_if_divergent_no_else_mod target = <gfx942> {
  func.func @test_if_divergent_no_else(
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

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = 1, output[32..63] = 2.
// Divergent condition, else branch, no results.
amdgcn.module @scf_if_divergent_with_else_mod target = <gfx942> {
  func.func @test_if_divergent_with_else(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %c1   = arith.constant 1 : i32
    %c2   = arith.constant 2 : i32
    %c4   = arith.constant 4 : i32
    %c32  = arith.constant 32 : i32
    %off  = arith.muli %tid, %c4 : i32
    %ptr  = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    %cond = arith.cmpi slt, %tid, %c32 : i32
    scf.if %cond {
      ptr.store %c1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    } else {
      ptr.store %c2, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..31] = tid, output[32..63] = 0.
// Divergent condition: lanes 0..31 store their tid; lanes 32..63 store 0.
amdgcn.module @scf_if_divergent_with_results_mod target = <gfx942> {
  func.func @test_if_divergent_with_results(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %c0   = arith.constant 0 : i32
    %c4   = arith.constant 4 : i32
    %c32  = arith.constant 32 : i32
    %off  = arith.muli %tid, %c4 : i32
    %ptr  = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    %cond = arith.cmpi slt, %tid, %c32 : i32
    // Then stores tid, else stores 0.
    scf.if %cond {
      ptr.store %tid, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    } else {
      ptr.store %c0, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[16..47] = 1, others = 0.
// Compound AND condition: (tid >= 16) AND (tid < 48).
amdgcn.module @scf_if_complex_condition_and_mod target = <gfx942> {
  func.func @test_if_complex_condition_and(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid    = aster_utils.thread_id x
    %c1     = arith.constant 1 : i32
    %c4     = arith.constant 4 : i32
    %c16    = arith.constant 16 : i32
    %c48    = arith.constant 48 : i32
    %off    = arith.muli %tid, %c4 : i32
    %ptr    = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    // (tid >= 16) AND (tid < 48).
    %cmp_lo = arith.cmpi sge, %tid, %c16 : i32
    %cmp_hi = arith.cmpi slt, %tid, %c48 : i32
    %cond   = arith.andi %cmp_lo, %cmp_hi : i1
    scf.if %cond {
      ptr.store %c1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }
    return
  }
}

// -----

// block=(64,1,1), grid=(1,1,1).
// output: 64 x i32 pre-filled with 0.
// Expected: output[0..15] = 1, output[16..47] = 0, output[48..63] = 1.
// Compound OR condition: (tid < 16) OR (tid >= 48).
amdgcn.module @scf_if_complex_condition_or_mod target = <gfx942> {
  func.func @test_if_complex_condition_or(
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid    = aster_utils.thread_id x
    %c1     = arith.constant 1 : i32
    %c4     = arith.constant 4 : i32
    %c16    = arith.constant 16 : i32
    %c48    = arith.constant 48 : i32
    %off    = arith.muli %tid, %c4 : i32
    %ptr    = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
    // (tid < 16) OR (tid >= 48).
    %cmp_lo = arith.cmpi slt, %tid, %c16 : i32
    %cmp_hi = arith.cmpi sge, %tid, %c48 : i32
    %cond   = arith.ori %cmp_lo, %cmp_hi : i1
    scf.if %cond {
      ptr.store %c1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }
    return
  }
}
