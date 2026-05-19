//===- AMDGPUOps.cpp - AMDGPU operations ------------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGPU/IR/AMDGPUOps.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amd_gpu;

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGPU/IR/AMDGPUOps.cpp.inc"
