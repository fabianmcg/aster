//===- AMDGPUAttrs.cpp - AMDGPU attributes ----------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGPU/IR/AMDGPUAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amd_gpu;

#include "aster/Dialect/AMDGPU/IR/AMDGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AMDGPU/IR/AMDGPUAttrs.cpp.inc"

void AMDGPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AMDGPU/IR/AMDGPUAttrs.cpp.inc"
      >();
}
