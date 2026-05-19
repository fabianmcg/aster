//===- AMDGPUOps.td - AMDGPU operations --------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGPU_IR_AMDGPUOPS_TD
#define ASTER_DIALECT_AMDGPU_IR_AMDGPUOPS_TD

include "aster/Dialect/AMDGPU/IR/AMDGPUDialect.td"
include "aster/Dialect/AMDGPU/IR/AMDGPUAttrs.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// BlockDimOp
//===----------------------------------------------------------------------===//

def AMDGPU_BlockDimOp : AMDGPU_Op<"block_dim", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AMDGPU block dimension operation";
  let description = [{
    Returns the block (workgroup) dimension in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// BlockIdOp
//===----------------------------------------------------------------------===//

def AMDGPU_BlockIdOp : AMDGPU_Op<"block_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AMDGPU block ID operation";
  let description = [{
    Returns the block (workgroup) ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// GridDimOp
//===----------------------------------------------------------------------===//

def AMDGPU_GridDimOp : AMDGPU_Op<"grid_dim", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AMDGPU grid dimension operation";
  let description = [{
    Returns the grid dimension in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

//===----------------------------------------------------------------------===//
// MakeBufferRsrcOp
//===----------------------------------------------------------------------===//

def AMDGPU_MakeBufferRsrcOp : AMDGPU_Op<"make_buffer_rsrc", [Pure]> {
  let summary = "High-level buffer resource descriptor constructor";
  let description = [{
    Constructs a buffer resource descriptor from a flat pointer, a record
    count, a stride, and packed flags. This is the high-level counterpart of
    `amdgcn.make_buffer_rsrc` and is intended to be lowered to that op during
    codegen.

    The result is a `!ptr.ptr` with the AMDGCN buffer address space so that
    subsequent `ptr.ptr_add` / `ptr.load` / `ptr.store` on it are lowered to
    buffer (MUBUF) instructions.

    Example:
    ```mlir
    %rsrc = amd_gpu.make_buffer_rsrc %ptr, %num_records, %stride,
              flags = 131072
            : (!ptr.ptr, i32, i32) -> !ptr.ptr<#amdgcn.addr_space<buffer, read_write>>
    ```
  }];
  let arguments = (ins
    AnyType:$ptr,
    I32:$num_records,
    I32:$stride,
    I32Attr:$flags
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ptr `,` $num_records `,` $stride `,`
    `flags` `=` $flags
    attr-dict `:` `(` type($ptr) `,` type($num_records) `,` type($stride) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def AMDGPU_ThreadIdOp : AMDGPU_Op<"thread_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
  let summary = "AMDGPU thread ID operation";
  let description = [{
    Returns the thread ID in the specified dimension (x, y, or z).
  }];
  let arguments = (ins DimAttr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = "$dim attr-dict";
}

#endif // ASTER_DIALECT_AMDGPU_IR_AMDGPUOPS_TD
