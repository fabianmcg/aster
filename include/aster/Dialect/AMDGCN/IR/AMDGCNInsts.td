//===- AMDGCNInsts.td - AMDGCN Instruction definitions -----*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INSTS_TD
#define AMDGCN_INSTS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VMem.td"

//===----------------------------------------------------------------------===//
// AMDGCN  Enums
//===----------------------------------------------------------------------===//

def EncodingEnum : AutoI32EnumCases<"AMDGCNEncoding",
  "AMDGCN encodings", !instances<Encoding>() # [
    PrettyEnumCaseInfo<"last_enc">
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def OpCodeEnum : AutoI32EnumCases<"AMDGCNOpCode",
  "AMDGCN op codes", !instances<AMDISAInstruction>() # [
    PrettyEnumCaseInfo<"last_op_code">
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

#endif // AMDGCN_INSTS_TD
