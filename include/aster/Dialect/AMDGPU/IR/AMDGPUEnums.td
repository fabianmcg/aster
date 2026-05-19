//===-- AMDGPUEnums.td - AMDGPU enumerations ---------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGPU_IR_AMDGPUENUMS_TD
#define ASTER_DIALECT_AMDGPU_IR_AMDGPUENUMS_TD

include "mlir/IR/Utils.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// GPU dimension identifier.
//===----------------------------------------------------------------------===//

def DimEnum : AutoI32EnumCases<"Dim",
  "GPU dimension identifier", [
    PrettyEnumCaseInfo<"x">,
    PrettyEnumCaseInfo<"y">,
    PrettyEnumCaseInfo<"z">,
  ]> {
  let cppNamespace = "::mlir::aster::amd_gpu";
}

#endif // ASTER_DIALECT_AMDGPU_IR_AMDGPUENUMS_TD
