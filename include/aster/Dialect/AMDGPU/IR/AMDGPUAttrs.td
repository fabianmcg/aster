//===- AMDGPUAttrs.td - AMDGPU attributes ------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGPU_IR_AMDGPUATTRS_TD
#define ASTER_DIALECT_AMDGPU_IR_AMDGPUATTRS_TD

include "aster/Dialect/AMDGPU/IR/AMDGPUDialect.td"
include "aster/Dialect/AMDGPU/IR/AMDGPUEnums.td"

//===----------------------------------------------------------------------===//
// GPU dimension identifier attribute.
//===----------------------------------------------------------------------===//

def DimAttr : AMDGPU_EnumAttr<"dim", DimEnum, []> {
  let description = [{
    AMDGPU gpu dimension identifier attribute.
  }];
}

#endif // ASTER_DIALECT_AMDGPU_IR_AMDGPUATTRS_TD
