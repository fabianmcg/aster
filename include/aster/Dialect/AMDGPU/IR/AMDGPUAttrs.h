//===- AMDGPUAttrs.h - AMDGPU dialect attributes ----------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGPU_IR_AMDGPUATTRS_H
#define ASTER_DIALECT_AMDGPU_IR_AMDGPUATTRS_H

#include "aster/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "aster/Dialect/AMDGPU/IR/AMDGPUEnums.h"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AMDGPU/IR/AMDGPUAttrs.h.inc"

#endif // ASTER_DIALECT_AMDGPU_IR_AMDGPUATTRS_H
