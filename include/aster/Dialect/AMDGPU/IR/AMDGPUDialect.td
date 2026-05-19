//===- AMDGPUDialect.td - AMDGPU dialect -------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGPU_IR_AMDGPUDIALECT_TD
#define ASTER_DIALECT_AMDGPU_IR_AMDGPUDIALECT_TD

include "mlir/IR/OpBase.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// AMDGPU dialect definition
//===----------------------------------------------------------------------===//

def AMDGPU_Dialect : Dialect {
  let name = "amd_gpu";
  let summary = "AMD GPU high-level dialect";
  let description = [{
    The AMDGPU dialect provides high-level AMD GPU operations for use before
    ISA-level codegen. Ops in this dialect are lowered to the amdgcn dialect
    during code generation.
  }];
  let cppNamespace = "::mlir::aster::amd_gpu";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    private:
      void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base class definitions
//===----------------------------------------------------------------------===//

class AMDGPU_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AMDGPU_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class AMDGPU_EnumAttr<string name, EnumInfo enumInfo, list<Trait> traits = []>
    : EnumAttr<AMDGPU_Dialect, enumInfo, name, traits> {
}

class AMDGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<AMDGPU_Dialect, mnemonic, traits>;

#endif // ASTER_DIALECT_AMDGPU_IR_AMDGPUDIALECT_TD
