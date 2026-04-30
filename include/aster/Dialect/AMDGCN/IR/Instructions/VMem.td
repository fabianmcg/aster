//===- VMem.td - AMDGCN VMem Instructions ------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the VMem instructions for AMDGCN.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_VMEM_TD
#define AMDGCN_VMEM_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

#ifndef GLOBALLOADINST_DEF
#define GLOBALLOADINST_DEF
class GlobalLoadInst<string mnemonic, list<Type> dstType> : AMDISAInstruction<mnemonic> {
  let archs = [CDNA3Isa, CDNA4Isa];
  let effects = [GlobalRead];
  let outputs = (ins
    AnyRegOf<dstType>:$dst
  );
  let inputs = (ins
    AnyRegOf<[VGPROf<1>, VGPROf<2>]>:$addr,
    OptionalOperand<TwoSGPROrVCCType>:$saddr
  );
  let trailingArgs = (ins
    OffsetType<13, /*signed=*/true>:$offset,
    NT:$nt,
    SC0:$sc0,
    SC1:$sc1
  );
  let asm_syntax = [
    ASMString<[ENC_FLAT_GLBL_CDNA3], "${mnemonic} $dst, $addr, off [offset] [sc0] [nt] [sc1]", CPred<"$_self.getSaddr() == nullptr">>,
    ASMString<[ENC_FLAT_GLBL_CDNA3], "${mnemonic} $dst, $addr, $saddr [offset] [sc0] [nt] [sc1]">,
    ASMString<[ENC_FLAT_GLBL_CDNA4], "${mnemonic} $dst, $addr, off [offset] [sc0] [sc1] [nt]", CPred<"$_self.getSaddr() == nullptr">>,
    ASMString<[ENC_FLAT_GLBL_CDNA4], "${mnemonic} $dst, $addr, $saddr [offset] [sc0] [sc1] [nt]">
  ];
  let props = [IsVMem, IsFlat, IsGlobal];
}
#endif // GLOBALLOADINST_DEF

#ifndef GLOBALLOADDWORD_DEF
#define GLOBALLOADDWORD_DEF
def GlobalLoadDword : GlobalLoadInst<"global_load_dword",
    /*dst=*/[VGPROrAGPROf<1>]
  > {
  let description = [{
    Load 32 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010100>];
}
#endif // GLOBALLOADDWORD_DEF


#endif // AMDGCN_VMEM_TD
