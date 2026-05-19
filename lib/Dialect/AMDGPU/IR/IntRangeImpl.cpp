//===- IntRangeImpl.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGPU/IR/AMDGPUOps.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amd_gpu;

static std::pair<ArrayRef<int32_t>, ArrayRef<int32_t>>
getLaunchBounds(Operation *op) {
  auto gpuFunc = op->getParentOfType<GPUFuncInterface>();
  if (!gpuFunc)
    return {};
  return {gpuFunc.getGridDims(), gpuFunc.getBlockDims()};
}

void BlockDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (blockDims.empty()) {
    return setResultRange(getResult(),
                          ConstantIntRanges::range(llvm::APInt(32, 0, false),
                                                   llvm::APInt(32, 1024, false),
                                                   false));
  }
  Dim dim = getDim();
  int32_t size =
      static_cast<int32_t>(blockDims.size()) > static_cast<int32_t>(dim)
          ? blockDims[static_cast<int32_t>(dim)]
          : 1;
  setResultRange(getResult(),
                 ConstantIntRanges::constant(llvm::APInt(32, size, false)));
}

void BlockIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (gridDims.empty()) {
    return setResultRange(
        getResult(),
        ConstantIntRanges::range(
            llvm::APInt(32, 0, false),
            llvm::APInt(32, std::numeric_limits<int32_t>::max() - 1, false),
            false));
  }
  Dim dim = getDim();
  int32_t size =
      static_cast<int32_t>(gridDims.size()) > static_cast<int32_t>(dim)
          ? gridDims[static_cast<int32_t>(dim)]
          : 1;
  // block_id ranges [0, gridDims[dim] - 1] (0-indexed).
  setResultRange(getResult(), ConstantIntRanges::range(
                                  llvm::APInt(32, 0, false),
                                  llvm::APInt(32, size - 1, false), false));
}

void GridDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (gridDims.empty()) {
    return setResultRange(
        getResult(),
        ConstantIntRanges::range(
            llvm::APInt(32, 0, false),
            llvm::APInt(32, std::numeric_limits<int32_t>::max(), false),
            false));
  }
  Dim dim = getDim();
  int32_t size =
      static_cast<int32_t>(gridDims.size()) > static_cast<int32_t>(dim)
          ? gridDims[static_cast<int32_t>(dim)]
          : 1;
  setResultRange(getResult(),
                 ConstantIntRanges::constant(llvm::APInt(32, size, false)));
}

void ThreadIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (blockDims.empty()) {
    return setResultRange(getResult(),
                          ConstantIntRanges::range(llvm::APInt(32, 0, false),
                                                   llvm::APInt(32, 1024, false),
                                                   false));
  }
  Dim dim = getDim();
  int32_t size =
      static_cast<int32_t>(blockDims.size()) > static_cast<int32_t>(dim)
          ? blockDims[static_cast<int32_t>(dim)]
          : 1;
  // thread_id ranges [0, blockDims[dim] - 1] (0-indexed).
  setResultRange(getResult(), ConstantIntRanges::range(
                                  llvm::APInt(32, 0, false),
                                  llvm::APInt(32, size - 1, false), false));
}
