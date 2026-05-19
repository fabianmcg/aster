//===- AMDGPUOps.h - AMDGPU dialect operations ------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGPU_IR_AMDGPUOPS_H
#define ASTER_DIALECT_AMDGPU_IR_AMDGPUOPS_H

#include "aster/Dialect/AMDGPU/IR/AMDGPUAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGPU/IR/AMDGPUOps.h.inc"

#endif // ASTER_DIALECT_AMDGPU_IR_AMDGPUOPS_H
