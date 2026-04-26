//===- Inst.td - Instruction base definitions --------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base classes and infrastructure for Aster instructions.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_INST_TD
#define ASTER_IR_INST_TD

include "aster/Interfaces/InstOpInterface.td"
include "aster/Support/EnumBase.td"
include "aster/Support/Utils.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Encodings and architectures
//===----------------------------------------------------------------------===//

// An architecture.
class Arch<string name> : EnumCaseInfo<name>;

// An instruction encoding.
class Encoding<string name> : PrettyEnumCaseInfo<name>;

// A concrete encoding of an instruction for a specific architecture.
class EncodedArch<Encoding _encoding, Arch _arch> {
  Arch arch = _arch;
  Encoding encoding = _encoding;
}

// A concrete instruction encoding with opcode.
class InstEnc<EncodedArch _arch, int _opcode> {
  EncodedArch arch = _arch;
  int opcode = _opcode;
}

//===----------------------------------------------------------------------===//
// Assembly syntax
//===----------------------------------------------------------------------===//

// Assembly syntax record carrying arch list, syntax string, and an optional predicate.
class ASMString<list<EncodedArch> _archs, string _syntax, Pred _pred = TruePred> {
  list<EncodedArch> archs = _archs;
  code syntax = _syntax;
  Pred pred = _pred;
}

//===----------------------------------------------------------------------===//
// Effects
//===----------------------------------------------------------------------===//

class Effect {
  // A string summary of the effect.
  string summary;
  // The code body for this effect. It has several special variables available:
  // - `$_op`, the current operation being processed.
  // - `$_effects`, the vector of effects collected for this instruction.
  code body;
}

//===----------------------------------------------------------------------===//
// Operands and modifiers
//===----------------------------------------------------------------------===//

// An input operand type.
class InType<list<Type> types> : AnyTypeOf<types, "an input operand">;

// An output operand type.
class OutType<list<Type> types> : AnyTypeOf<types, "an output operand">;

// An input/output operand type.
class InOutType<list<Type> types> : AnyTypeOf<types, "an input/output operand">;

// A type that appears in a particular encoding of an instruction.
class EncType<list<EncodedArch> _archs, Type _type> : TypeAlias<_type> {
  list<EncodedArch> archs = _archs;
}

// A modifier operand type.
class ModType<Type _type> : TypeAlias<_type>;

// A modifier attribute.
class ModAttr<Attr _attr, string _default = ""> : DefaultValuedAttr<_attr, _default> {
}

class UnitModifier<string name, string summary> : ModAttr<UnitAttr, "false">;

//===----------------------------------------------------------------------===//
// Instruction base class
//===----------------------------------------------------------------------===//

/// Helper class to represent an argument.
class InstOperand<string _n, TypeConstraint _t> {
  string name = _n;
  TypeConstraint type = _t;

  // Extract if the argument is optional.
  bit isOptional = !isa<Optional>(type);

  // Extract if the argument is variadic.
  bit isVariadic = !isa<Variadic>(type);

  // Extract if the argument is a variadic of a variadic.
  bit isVariadicOfVariadic = !isa<VariadicOfVariadic>(type);

  // Extract if the argument is optional, variadic, or a variadic of a variadic.
  bit isOpt = !or(isOptional, isVariadic, isVariadicOfVariadic);
}

/// Helper class to get the optional type of a type.
class _GetOptionalType<TypeConstraint type, bit isOpt> {
  TypeConstraint optType = !if(isOpt, type, Optional<
    !if(!isa<Type>(type),
      !cast<Type>(type), Type<type.predicate, type.summary, type.cppType>)
  >);
}

/// Helper class to extract information from a list of dag arguments.
class InstOperandInfo<dag args> {
  // Generate indices for the args.
  list<int> indices = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!range(!size(args))
  );

  // Get the number of arguments.
  int numArgs = !size(indices);

  // Extract the names of the arguments.
  list<string> names = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, !getdagname(args, i))
  );

  // Extract the types of the arguments.
  list<TypeConstraint> types = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, !getdagarg<TypeConstraint>(args, i))
  );

  // Create argument objects.
  list<InstOperand> argList = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, InstOperand<names[i], types[i]>)
  );

  // Create argument objects with optional types.
  list<TypeConstraint> typesAsOpt = !if(!empty(argList),
    /*true_value=*/[],
    /*false_value=*/!foreach(arg, argList, _GetOptionalType<arg.type, arg.isOpt>.optType)
  );

  // Extract if the arguments are optional.
  list<int> isOpt = !if(!empty(argList),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, argList[i].isOpt)
  );

  // Get the number of optional arguments.
  int numOptArgs = !foldl(0, isOpt, acc, v, !add(v, acc));
}

/// Helper class to extract information from a dag.
class InstructionDagInfo<dag args> {
  // Generate indices for the args.
  list<int> indices = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!range(!size(args))
  );

  // Extract the types of the arguments.
  list<bit> isOperand = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, !initialized(!getdagarg<TypeConstraint>(args, i)))
  );

  // The number of leading operands.
  int numOperands = !if(!empty(args),
    /*true_value=*/0,
    /*false_value=*/!foldl(0, isOperand, acc, v, !add(!if(v, 1, 0), acc))
  );
}

/// Helper class to extract information from a list of dag arguments.
class InstructionInfo<dag inputs, dag outputs, dag modifiers> {
  InstOperandInfo inputsInfo = InstOperandInfo<inputs>;
  InstOperandInfo outputsInfo = InstOperandInfo<outputs>;

  // Get the number of arguments.
  int numArgs = !add(inputsInfo.numArgs, outputsInfo.numArgs);

  // Get the number of optional arguments.
  int numOptArgs = !add(inputsInfo.numOptArgs, outputsInfo.numOptArgs);

  // Determine if the instruction requires attr-sized operand segments.
  bit requiresAttrSizedOperandSegments = !gt(numOptArgs, 1);

  // Determine if the instruction requires attr-sized result segments.
  bit requiresAttrSizedResultSegments = !gt(outputsInfo.numArgs, 1);

  // Generate the traits for the instruction.
  list<Trait> traits = !listconcat([],
    !if(requiresAttrSizedOperandSegments, [AttrSizedOperandSegments], []),
    !if(requiresAttrSizedResultSegments, [AttrSizedResultSegments], [])
  );

  // Generate result names by appending "_res" to argument names.
  list<string> resNames = !foreach(n, outputsInfo.names, n#"_res");
  dag instResults = !if(!gt(outputsInfo.numArgs, 0),
    /*true_value=*/!dag(outs, outputsInfo.typesAsOpt, resNames),
    /*false_value=*/(outs)
  );

  // The number of modifier arguments.
  int numModifiers = InstructionDagInfo<modifiers>.numOperands;

  // The number of instruction results.
  int numInstResults = !size(instResults);
}

class Instruction<Dialect dialect, string mnemonic, list<Trait> opTraits = []>
    : Op<dialect, mnemonic, opTraits> {
  // A list of architectures this instruction is valid for, if empty, then
  // it's valid for all architectures.
  list<Arch> archs = [];
  // A list of encodings for this instruction.
  list<InstEnc> encodings = [];
  // A list of side effects of this instruction.
  list<Effect> effects = [];
  // A list of assembly syntaxes for this instruction.
  list<ASMString> asm_syntax = [];
  // The outputs of the instruction.
  dag outputs = (ins);
  // The inputs of the instruction.
  dag inputs = (ins);
  // The modifiers of the instruction.
  dag modifiers = (ins);
  // The instruction operand and result information.
  InstructionInfo info = InstructionInfo<inputs, outputs, modifiers>;
  // The arguments of the MLIR operation.
  let arguments = !con(outputs, inputs, modifiers);
  // Helper with the results of the instruction.
  dag instResults = info.instResults;
  // The results of the operation.
  let results = instResults;
  let traits = [
    DeclareOpInterfaceMethods<InstOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ] # opTraits;
  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

class CPPType<string cppType, string summary> : Type<TruePred, summary, cppType> {
  let predicate = CPred<
    StrSubst<"::llvm::isa<$_type>($_self)", [VarRepl<"_type", cppType>]>.result
  >;
  let description = summary;
}

#endif // ASTER_IR_INST_TD
