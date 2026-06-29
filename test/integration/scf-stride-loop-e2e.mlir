// RUN: aster-opt %s --verify-roundtrip --split-input-file
//
// E2E tests for scf.for (stride-loop) + scf.if control-flow lowering.
//
// Five kernels, each in its own module (block=(128,1,1), grid=(1,1,1)):
//   test_stride_loop             -- for (i=tid; i<n; i+=128) x[i]=i.
//   test_stride_loop_if_else     -- store i+1 if i%2==0, else 2*i+1.
//   test_stride_loop_if_only     -- store i+1 only if i%2==0.
//   test_stride_loop_else_only   -- store 2*i+1 only if i%2!=0.
//   test_stride_loop_compound_and -- store 6*i+1 if i%2==0&&i%3==0, else i.

// -----

// block=(128,1,1), grid=(1,1,1).
// Input: scalar n (i32); output: n x i32 pre-filled with -1.
// Expected: x[i] = i for all i in [0, n).
amdgcn.module @scf_stride_loop_mod target = <gfx942> {
  func.func @test_stride_loop(
      %n: i32,
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %bdim = aster_utils.block_dim x
    %c4   = arith.constant 4 : i32
    scf.for unsigned %i = %tid to %n step %bdim : i32 {
      %off = arith.muli %i, %c4 : i32
      %ptr = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
      ptr.store %i, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
    }
    return
  }
}

// -----

// block=(128,1,1), grid=(1,1,1).
// Input: scalar n (i32); output: n x i32 pre-filled with -1.
// Expected: x[i] = i+1 if i%2==0, else 2*i+1, for i in [0, n).
amdgcn.module @scf_stride_loop_if_else_mod target = <gfx942> {
  func.func @test_stride_loop_if_else(
      %n: i32,
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %bdim = aster_utils.block_dim x
    %c0   = arith.constant 0 : i32
    %c1   = arith.constant 1 : i32
    %c2   = arith.constant 2 : i32
    %c4   = arith.constant 4 : i32
    scf.for unsigned %i = %tid to %n step %bdim : i32 {
      %off  = arith.muli %i, %c4 : i32
      %ptr  = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
      // Divergent condition: i%2==0.
      %r2   = arith.remsi %i, %c2 : i32
      %even = arith.cmpi eq, %r2, %c0 : i32
      scf.if %even {
        // Store i+1.
        %ip1 = arith.addi %i, %c1 : i32
        ptr.store %ip1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      } else {
        // Store 2*i+1.
        %i2   = arith.muli %i, %c2 : i32
        %i2p1 = arith.addi %i2, %c1 : i32
        ptr.store %i2p1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      }
    }
    return
  }
}

// -----

// block=(128,1,1), grid=(1,1,1).
// Input: scalar n (i32); output: n x i32 pre-filled with -1.
// Expected: x[i] = i+1 if i%2==0 and i<n, else -1.
amdgcn.module @scf_stride_loop_if_only_mod target = <gfx942> {
  func.func @test_stride_loop_if_only(
      %n: i32,
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %bdim = aster_utils.block_dim x
    %c0   = arith.constant 0 : i32
    %c1   = arith.constant 1 : i32
    %c2   = arith.constant 2 : i32
    %c4   = arith.constant 4 : i32
    scf.for unsigned %i = %tid to %n step %bdim : i32 {
      %off  = arith.muli %i, %c4 : i32
      %ptr  = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
      %r2   = arith.remsi %i, %c2 : i32
      %even = arith.cmpi eq, %r2, %c0 : i32
      scf.if %even {
        %ip1 = arith.addi %i, %c1 : i32
        ptr.store %ip1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      }
    }
    return
  }
}

// -----

// block=(128,1,1), grid=(1,1,1).
// Input: scalar n (i32); output: n x i32 pre-filled with -1.
// Expected: x[i] = 2*i+1 if i%2!=0 and i<n, else -1.
amdgcn.module @scf_stride_loop_else_only_mod target = <gfx942> {
  func.func @test_stride_loop_else_only(
      %n: i32,
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %bdim = aster_utils.block_dim x
    %c0   = arith.constant 0 : i32
    %c1   = arith.constant 1 : i32
    %c2   = arith.constant 2 : i32
    %c4   = arith.constant 4 : i32
    scf.for unsigned %i = %tid to %n step %bdim : i32 {
      %off = arith.muli %i, %c4 : i32
      %ptr = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
      %r2  = arith.remsi %i, %c2 : i32
      %odd = arith.cmpi ne, %r2, %c0 : i32
      scf.if %odd {
        %i2   = arith.muli %i, %c2 : i32
        %i2p1 = arith.addi %i2, %c1 : i32
        ptr.store %i2p1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      }
    }
    return
  }
}

// -----

// block=(128,1,1), grid=(1,1,1).
// Input: scalar n (i32); output: n x i32 pre-filled with -1.
// Expected: x[i] = 6*i+1 if i%2==0&&i%3==0, else x[i]=i, for i in [0, n).
amdgcn.module @scf_stride_loop_compound_and_mod target = <gfx942> {
  func.func @test_stride_loop_compound_and(
      %n: i32,
      %buf: !ptr.ptr<#amdgcn.addr_space<global, read_write>>)
      attributes {gpu.kernel} {
    %tid  = aster_utils.thread_id x
    %bdim = aster_utils.block_dim x
    %c0   = arith.constant 0 : i32
    %c1   = arith.constant 1 : i32
    %c2   = arith.constant 2 : i32
    %c3   = arith.constant 3 : i32
    %c4   = arith.constant 4 : i32
    %c6   = arith.constant 6 : i32
    scf.for unsigned %i = %tid to %n step %bdim : i32 {
      %off  = arith.muli %i, %c4 : i32
      %ptr  = ptr.ptr_add %buf, %off : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, i32
      // Compound condition: i%2==0 && i%3==0.
      %r2   = arith.remsi %i, %c2 : i32
      %even = arith.cmpi eq, %r2, %c0 : i32
      %r3   = arith.remsi %i, %c3 : i32
      %div3 = arith.cmpi eq, %r3, %c0 : i32
      %both = arith.andi %even, %div3 : i1
      scf.if %both {
        // Store 6*i+1.
        %i6   = arith.muli %i, %c6 : i32
        %i6p1 = arith.addi %i6, %c1 : i32
        ptr.store %i6p1, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      } else {
        // Store i.
        ptr.store %i, %ptr : i32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
      }
    }
    return
  }
}
