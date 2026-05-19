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
