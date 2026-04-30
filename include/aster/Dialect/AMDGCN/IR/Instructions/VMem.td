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

#ifndef GLOBALLOADDWORDX2_DEF
#define GLOBALLOADDWORDX2_DEF
def GlobalLoadDwordx2 : GlobalLoadInst<"global_load_dwordx2",
    /*dst=*/[VGPROrAGPROf<2>]
  > {
  let description = [{
    Load 64 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010101>];
}
#endif // GLOBALLOADDWORDX2_DEF

#ifndef GLOBALLOADDWORDX3_DEF
#define GLOBALLOADDWORDX3_DEF
def GlobalLoadDwordx3 : GlobalLoadInst<"global_load_dwordx3",
    /*dst=*/[VGPROrAGPROf<3>]
  > {
  let description = [{
    Load 96 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010110>];
}
#endif // GLOBALLOADDWORDX3_DEF

#ifndef GLOBALLOADDWORDX4_DEF
#define GLOBALLOADDWORDX4_DEF
def GlobalLoadDwordx4 : GlobalLoadInst<"global_load_dwordx4",
    /*dst=*/[VGPROrAGPROf<4>]
  > {
  let description = [{
    Load 128 bits of data from the global aperture into a vector register.
  }];
  let encodings = [InstEnc<[ENC_FLAT_GLBL_CDNA3, ENC_FLAT_GLBL_CDNA4], 0b0010111>];
}
#endif // GLOBALLOADDWORDX4_DEF


#ifndef VOP2OUT1IN2INST_DEF
#define VOP2OUT1IN2INST_DEF
class VOP2Out1In2Inst<string mnemonic, list<Type> dstType, list<Type> src0Type, list<Type> src1Type> : AMDISAInstruction<mnemonic> {
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dstType>:$dst
  );
  let inputs = (ins
    AnyOperandOf<src0Type>:$src0,
    AnyOperandOf<src1Type>:$src1
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4, ENC_VOP3_CDNA4], "${mnemonic} $dst, $src0, $src1">,
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64 $dst, $src0, $src1">
  ];
}
#endif // VOP2OUT1IN2INST_DEF

#ifndef VADDU32_DEF
#define VADDU32_DEF
def VAddU32D : VOP2Out1In2Inst<"v_add_u32",
    /*dst=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Add two unsigned 32-bit integer inputs and store the result into a vector
    register. No carry-in or carry-out support.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b110100, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100110100, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VADDU32_DEF

#endif // AMDGCN_VMEM_TD
