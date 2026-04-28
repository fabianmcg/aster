//===- Inst.td - AMDGCN Instruction base -------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the base file for AMDGCN instructions.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_TD
#define AMDGCN_INST_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/IR/Inst.td"

//===----------------------------------------------------------------------===//
// Base instruction class for AMDGCN instructions.
//===----------------------------------------------------------------------===//

class AMDISAInstruction<string mnemonic, list<Trait> opTraits = []>
    : Instruction<AMDGCN_Dialect, mnemonic, opTraits>;

//===----------------------------------------------------------------------===//
// AMDGCN supported architectures.
//===----------------------------------------------------------------------===//

def CDNA3 : Arch<"CDNA3">;
def CDNA4 : Arch<"CDNA4">;

//===----------------------------------------------------------------------===//
// Encoding name records
//===----------------------------------------------------------------------===//

def ENC_DS : Encoding<"ds">;
def ENC_FLAT_GLBL : Encoding<"flat_glbl">;
def ENC_MUBUF : Encoding<"mubuf">;
def ENC_SMEM : Encoding<"smem">;
def ENC_SOP1 : Encoding<"sop1">;
def ENC_SOP2 : Encoding<"sop2">;
def ENC_SOPC : Encoding<"sopc">;
def ENC_SOPK : Encoding<"sopk">;
def ENC_SOPP : Encoding<"sopp">;
def ENC_VOP1 : Encoding<"vop1">;
def ENC_VOP2 : Encoding<"vop2">;
def ENC_VOP3 : Encoding<"vop3">;
def ENC_VOP3P : Encoding<"vop3p">;
def ENC_VOP3PX2 : Encoding<"vop3px2">;
def ENC_VOPC : Encoding<"vopc">;
def VOP3P_MFMA : Encoding<"vop3p_mfma">;

//===----------------------------------------------------------------------===//
// EncodedArch records
//===----------------------------------------------------------------------===//

def ENC_DS_CDNA3 : EncodedArch<ENC_DS, CDNA3>;
def ENC_DS_CDNA4 : EncodedArch<ENC_DS, CDNA4>;
def ENC_FLAT_GLBL_CDNA3 : EncodedArch<ENC_FLAT_GLBL, CDNA3>;
def ENC_FLAT_GLBL_CDNA4 : EncodedArch<ENC_FLAT_GLBL, CDNA4>;
def ENC_MUBUF_CDNA3 : EncodedArch<ENC_MUBUF, CDNA3>;
def ENC_MUBUF_CDNA4 : EncodedArch<ENC_MUBUF, CDNA4>;
def ENC_SMEM_CDNA3 : EncodedArch<ENC_SMEM, CDNA3>;
def ENC_SMEM_CDNA4 : EncodedArch<ENC_SMEM, CDNA4>;
def ENC_SOP1_CDNA3 : EncodedArch<ENC_SOP1, CDNA3>;
def ENC_SOP1_CDNA4 : EncodedArch<ENC_SOP1, CDNA4>;
def ENC_SOP2_CDNA3 : EncodedArch<ENC_SOP2, CDNA3>;
def ENC_SOP2_CDNA4 : EncodedArch<ENC_SOP2, CDNA4>;
def ENC_SOPC_CDNA3 : EncodedArch<ENC_SOPC, CDNA3>;
def ENC_SOPC_CDNA4 : EncodedArch<ENC_SOPC, CDNA4>;
def ENC_SOPK_CDNA3 : EncodedArch<ENC_SOPK, CDNA3>;
def ENC_SOPK_CDNA4 : EncodedArch<ENC_SOPK, CDNA4>;
def ENC_SOPP_CDNA3 : EncodedArch<ENC_SOPP, CDNA3>;
def ENC_SOPP_CDNA4 : EncodedArch<ENC_SOPP, CDNA4>;
def ENC_VOP1_CDNA3 : EncodedArch<ENC_VOP1, CDNA3>;
def ENC_VOP1_CDNA4 : EncodedArch<ENC_VOP1, CDNA4>;
def ENC_VOP2_CDNA3 : EncodedArch<ENC_VOP2, CDNA3>;
def ENC_VOP2_CDNA4 : EncodedArch<ENC_VOP2, CDNA4>;
def ENC_VOP3PX2_CDNA4 : EncodedArch<ENC_VOP3PX2, CDNA4>;
def ENC_VOP3P_CDNA3 : EncodedArch<ENC_VOP3P, CDNA3>;
def ENC_VOP3P_CDNA4 : EncodedArch<ENC_VOP3P, CDNA4>;
def ENC_VOP3_CDNA3 : EncodedArch<ENC_VOP3, CDNA3>;
def ENC_VOP3_CDNA4 : EncodedArch<ENC_VOP3, CDNA4>;
def ENC_VOPC_CDNA3 : EncodedArch<ENC_VOPC, CDNA3>;
def ENC_VOPC_CDNA4 : EncodedArch<ENC_VOPC, CDNA4>;
def VOP3P_MFMA_CDNA3 : EncodedArch<VOP3P_MFMA, CDNA3>;
def VOP3P_MFMA_CDNA4 : EncodedArch<VOP3P_MFMA, CDNA4>;

//===----------------------------------------------------------------------===//
// Modifier attribute constraints
//===----------------------------------------------------------------------===//

def GDS : UnitModifier<"GDS", "1=GDS, 0=LDS">;
def GLC : UnitModifier<"GLC", "If set, operation is globally coherent.">;
def IDXEN : UnitModifier<"IDXEN", "If set, send VADDR as an index. If unset, treat the index as zero.">;
def LDS : UnitModifier<"LDS", "If set, data is written to LDS memory. If unset, data is read from/written to a VGPR.">;
def NT : UnitModifier<"NT", "Non-Temporal: data is streamed or uncached.">;
def OFFEN : UnitModifier<"OFFEN", "If set, send VADDR as an offset. If unset, send the instruction offset stored in OFFSET. Only one of these offsets may be sent.">;
def SC0 : UnitModifier<"SC0", "For regular load and store operations, SC0 and SC1 indicate memory scope such as wave, group, device and system. For atomics, SC0 means Atomic with return. 1 = Atomic with return, 0 = Atomic with no return.">;
def SC1 : UnitModifier<"SC1", "For regular load and store operations, SC0 and SC1 indicate memory scope such as wave, group, device and system.">;

//===----------------------------------------------------------------------===//
// Inline literal constraints.
//===----------------------------------------------------------------------===//

class InlineLitFP<list<Type> types> : AnyTypeOf<types, "an inline literal floating-point operand", "mlir::FloatType">, ConstValueTypeConstraint {
  let constraint = CPred<[{
    ::mlir::aster::checkFloatConst($_self, $_strictness, {-4.0, -2.0, -1.0, -0.5, 0.15915494, 0.5, 1.0, 2.0, 4.0})
  }]>;
}

class InlineLitInt<list<Type> types> : AnyTypeOf<types, "an inline literal integer operand", "mlir::IntegerType">, ConstValueTypeConstraint {
  list<int> values = [-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];
  let constraint = CPred<StrSubst<[{
    ::mlir::aster::checkIntConst($_self, $_strictness, $_values)
  }], [VarRepl<"_value", "{" # !interleave(values, ", ") # "}">]>.result>;
}

class InlineLitIntPos<list<Type> types> : AnyTypeOf<types, "an inline literal positive integer operand", "mlir::IntegerType">, ConstValueTypeConstraint {
  list<int> values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];
  let constraint = CPred<StrSubst<[{
    ::mlir::aster::checkIntConst($_self, $_strictness, $_values)
  }], [VarRepl<"_value", "{" # !interleave(values, ", ") # "}">]>.result>;
}

//===----------------------------------------------------------------------===//
// Integer type constraint.
//===----------------------------------------------------------------------===//

class IntType<int bitWidth> : I<bitWidth>;

class OffsetType<int bitWidth, bit _isSigned> : I<32>, ConstValueTypeConstraint {
  int offsetWidth = bitWidth;
  bit isSigned = _isSigned;
  let constraint = CPred<StrSubst<[{
    ::mlir::aster::checkOffsetConst($_self, $_strictness, $_offsetWidth)
  }], [IntRepl<"_offsetWidth", offsetWidth>]>.result>;
}

//===----------------------------------------------------------------------===//
// Operand types
//===----------------------------------------------------------------------===//

class AMDReg<string cppType, string name> :
  CPPType<"::mlir::aster::amdgcn::" # cppType, "register of type: " # name> {
}

def AGPRType : AMDReg<"AGPRType", "AGPR">;
def VGPRType : AMDReg<"VGPRType", "VGPR">;
def SGPRType : AMDReg<"SGPRType", "SGPR">;

def VCCType : AMDReg<"VCCType", "VCC">;
def VCCLoType : AMDReg<"VCCLoType", "VCC Lo">;
def VCCHiType : AMDReg<"VCCHiType", "VCC Hi">;
def M0Type : AMDReg<"M0Type", "M0">;
def EXECType : AMDReg<"EXECType", "EXEC">;
def EXECLoType : AMDReg<"EXECLoType", "EXEC Lo">;
def EXECHiType : AMDReg<"EXECHiType", "EXEC Hi">;
def VCCZType : AMDReg<"VCCZType", "VCCZ">;
def EXECZType : AMDReg<"EXECZType", "EXECZ">;
def SCCType : AMDReg<"SCCType", "SCC">;

class AnyRegOf<list<Type> types> : AnyTypeOf<types, "a register operand", "mlir::aster::RegisterTypeInterface">;

class SizedGPRRange<list<Type> types, list<int> size> :
  ConfinedType<AnyRegOf<types>, [
    CPred<"::mlir::aster::amdgcn::hasSize($_self, {" # !interleave(size, ", ") # "})">
  ], "sized GPR range"> {
}

class AGPROf<int numRegs> : SizedGPRRange<[AGPRType], [numRegs]>;
class SGPROf<int numRegs> : SizedGPRRange<[SGPRType], [numRegs]>;
class VGPROf<int numRegs> : SizedGPRRange<[VGPRType], [numRegs]>;

class OneGPROrSRegOrConstOf<Type litType, Type dataType> : AnyTypeOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, litType, dataType]>;
class OneGPROrSRegOrLitOf<Type litType> : AnyRegOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, litType]>;
class OneGPROrVCCM0OrFPConstOf<Type floatLit, Type fpConst> : AnyTypeOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, VCCZType, EXECZType, SCCType, floatLit, fpConst]>;
class OneGPROrVCCM0OrFPLitOf<Type floatLit> : AnyTypeOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, VCCZType, EXECZType, SCCType, floatLit]>;
class OneSGPROrSRegOrIntConstOf<Type intLit, Type intConst> : AnyTypeOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, intLit, intConst]>;
class OneSGPROrSRegOrLitConstOf<Type floatLit, Type intLit, Type fpConst, Type intConst> : AnyTypeOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, floatLit, intLit, fpConst, intConst]>;
class TwoGPROrSRegOrConstOf<Type litType, Type intConst> : AnyRegOf<[VGPROf<2>, SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, litType, intConst]>;
class TwoGPROrVCCOrFPLitOf<Type floatLit> : AnyTypeOf<[VGPROf<2>, SGPROf<2>, VCCType, VCCZType, EXECZType, SCCType, floatLit]>;
class VGPROrAGPROf<int numRegs> : AnyRegOf<[VGPROf<numRegs>, AGPROf<numRegs>]>;
class VGPROrAGPROrLitOf<int numRegs, Type litType> : AnyRegOf<[VGPROf<numRegs>, AGPROf<numRegs>, litType]>;

def OneGPROrSRegOrConstType : AnyTypeOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, IntType<32>, InlineLitInt<[IntType<32>]>, InlineLitFP<[F32]>, F32]>;
def OneSGPROrSRegType : AnyRegOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType]>;
def OneSGPROrVCCM0OrIntLitType : AnyTypeOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, InlineLitIntPos<[IntType<32>]>]>;
def OneSGPROrVCCM0Type : AnyRegOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type]>;
def OneSGPROrVCCType : AnyRegOf<[SGPROf<1>, VCCLoType, VCCHiType]>;
def TwoGPROrSRegOrIntLitType : AnyTypeOf<[VGPROf<2>, SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, InlineLitInt<[IntType<64>]>]>;
def TwoGPROrVCCOrFPConstType : AnyTypeOf<[VGPROf<2>, SGPROf<2>, VCCType, VCCZType, EXECZType, SCCType, F64, InlineLitFP<[F64]>]>;
def TwoSGPROrSRegOrConstType : AnyTypeOf<[SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, IntType<64>, InlineLitInt<[IntType<64>]>, InlineLitFP<[F64]>, F64]>;
def TwoSGPROrSRegOrIntConstType : AnyTypeOf<[SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, IntType<64>, InlineLitInt<[IntType<64>]>]>;
def TwoSGPROrSRegOrLitType : AnyTypeOf<[SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, InlineLitInt<[IntType<64>]>, InlineLitFP<[F64]>]>;
def TwoSGPROrSRegType : AnyRegOf<[SGPROf<2>, VCCType, EXECType]>;
def TwoSGPROrVCCType : AnyRegOf<[SGPROf<2>, VCCType]>;

//===----------------------------------------------------------------------===//
// Effects
//===----------------------------------------------------------------------===//

def GlobalRead : Effect {
  let summary = "Reads from global memory.";
  let body = [{}];
}

#endif // AMDGCN_INST_TD
