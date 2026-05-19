//===- AMDGPU.cpp - AMDGPU dialect ------------------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGPU/IR/AMDGPUOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amd_gpu;

//===----------------------------------------------------------------------===//
// AMDGPU dialect
//===----------------------------------------------------------------------===//

void AMDGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/AMDGPU/IR/AMDGPUOps.cpp.inc"
      >();
  registerAttributes();
}

#include "aster/Dialect/AMDGPU/IR/AMDGPUDialect.cpp.inc"
