//===- InstMethodsGen.cpp -------------------------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates the AMDGCN instruction method definitions.
//
//===----------------------------------------------------------------------===//

#include "InstCommon.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// TableGen class for ISA-level instruction ops (different from AMDInst which
/// is the instruction metadata class).
static constexpr StringRef AMDISAInstClassType = "AMDISAInstruction";

/// Returns the raw adaptor accessor expression for an operand.
static std::string getAdaptorAccessor(StringRef name) {
  if (name.empty())
    return "adaptor";
  return "adaptor.get" +
         llvm::convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true) +
         "()";
}

/// Returns the constraint for an operand, unwrapping Optional<...>.
/// Returns std::nullopt when an Optional operand has no resolvable baseType.
static std::optional<std::pair<mlir::tblgen::Constraint, bool>>
resolveConstraint(const mlir::tblgen::NamedTypeConstraint &operand) {
  const mlir::tblgen::TypeConstraint &tc = operand.constraint;
  const llvm::Record *rec = &tc.getDef();
  bool isOptional = false;
  if (tc.isOptional()) {
    rec = rec->getValueAsOptionalDef("baseType");
    if (!rec)
      return std::nullopt;
    isOptional = true;
  }
  return std::pair{mlir::tblgen::TypeConstraint(rec), isOptional};
}

/// Emits a single constraint check block. `selfValue` is the raw adaptor
/// accessor expression (e.g. `adaptor.getFoo()`). The block declares
/// `_selfValue` (raw) and `_self` (type-or-value wrapped) locals, then checks
/// the condition template.
static void emitConstraintCheck(raw_ostream &os, mlir::tblgen::FmtContext &ctx,
                                StringRef selfValue,
                                const mlir::tblgen::Constraint &constraint,
                                bool isOptional, StringRef failureExpr) {
  const std::string_view body = R"(  {
    auto &&_selfValue = $0;
    auto &&_self = getTypeOrValue($0);
    (void)_self;
    (void)_selfValue;
    if ($1!($2))
      return $3;
  })";
  ctx.withSelf("_self");
  ctx.addSubst("_selfValue", "_selfValue");
  StringRef optionalStr = isOptional ? "_self && " : "";
  StrStream stream;
  stream.os << mlir::tblgen::tgfmt(body.data(), &ctx,
                                   /*0=*/selfValue,
                                   /*1=*/optionalStr,
                                   /*2=*/constraint.getConditionTemplate(),
                                   /*3=*/failureExpr);
  os << mlir::tblgen::tgfmt(stream.str.data(), &ctx);
}

/// Parses the `constraints` dag of an encoding into a name -> Type record map.
/// Emits a fatal error if a name does not match any operand of `op`.
static llvm::StringMap<const llvm::Record *>
parseEncodingOverrides(const InstEncRecord &enc,
                       const mlir::tblgen::Operator &op) {
  llvm::StringMap<const llvm::Record *> result;
  const llvm::DagInit *dag = enc.getConstraintsDag();
  for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i) {
    StringRef name = dag->getArgNameStr(i);
    const auto *defInit = dyn_cast<llvm::DefInit>(dag->getArg(i));
    if (!defInit)
      llvm::PrintFatalError(enc.getDef().getLoc(),
                            "encoding constraint argument '" + name +
                                "' must be a Type record");
    bool found = false;
    for (int j = 0, je = op.getNumOperands(); j < je; ++j) {
      if (op.getOperand(j).name != name)
        continue;
      found = true;
      break;
    }
    if (!found)
      llvm::PrintFatalError(
          enc.getDef().getLoc(),
          "encoding constraint references unknown argument '" + name +
              "' in op '" + op.getOperationName() + "'");
    result[name] = defInit->getDef();
  }
  return result;
}

/// Collects the set of operand names that have per-encoding overrides in any
/// encoding.
static llvm::StringSet<>
collectOverriddenNames(llvm::ArrayRef<InstEncRecord> encodings) {
  llvm::StringSet<> overriddenNames;
  for (const InstEncRecord &enc : encodings) {
    const llvm::DagInit *dag = enc.getConstraintsDag();
    for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i)
      overriddenNames.insert(dag->getArgNameStr(i));
  }
  return overriddenNames;
}

//===----------------------------------------------------------------------===//
// Encoding dispatch helpers
//===----------------------------------------------------------------------===//

/// A flattened (arch, encoding) pair used for dispatch iteration.
struct ArchEncPair {
  StringRef archId;
  StringRef encodingId;
  InstEncRecord enc;
};

/// Builds a flat list of (arch, encoding) pairs from the encoding list,
/// grouped by arch in encounter order.
static llvm::SmallVector<ArchEncPair>
flattenEncodings(llvm::ArrayRef<InstEncRecord> encodings) {
  llvm::SmallVector<ArchEncPair> pairs;
  for (const InstEncRecord &enc : encodings)
    for (const EncodedArchRecord &ea : enc.getEncodedArchs())
      pairs.push_back({ea.getArch().getIdentifier(),
                       ea.getEncoding().getIdentifier(), enc});
  return pairs;
}

/// Returns the distinct arch IDs in encounter order.
static llvm::SmallVector<StringRef>
getArchOrder(llvm::ArrayRef<ArchEncPair> pairs) {
  llvm::SmallVector<StringRef> order;
  llvm::StringSet<> seen;
  for (const ArchEncPair &p : pairs) {
    if (seen.insert(p.archId).second)
      order.push_back(p.archId);
  }
  return order;
}

//===----------------------------------------------------------------------===//
// isValid generation (split into sub-emitters)
//===----------------------------------------------------------------------===//

/// Emits the `isValidPair` lambda that checks whether the (arch, encoding)
/// combination is valid for this instruction.
static void emitArchEncodingCheck(llvm::ArrayRef<ArchEncPair> pairs,
                                  llvm::ArrayRef<StringRef> archOrder,
                                  mlir::tblgen::FmtContext &ctx,
                                  raw_ostream &os) {
  os << "  auto isValidPair = [&]() -> bool {\n";
  for (StringRef archId : archOrder) {
    ctx.addSubst("_archId", archId);
    os << mlir::tblgen::tgfmt(
        R"(    if (tgt.getTargetFamily() == ::mlir::aster::amdgcn::ISAVersion::$_archId)
      return )",
        &ctx);
    llvm::interleave(
        llvm::make_filter_range(
            pairs, [&](const ArchEncPair &p) { return p.archId == archId; }),
        os,
        [&](const ArchEncPair &p) {
          ctx.addSubst("_encodingId", p.encodingId);
          os << mlir::tblgen::tgfmt(
              "encoding == ::mlir::aster::amdgcn::AMDGCNEncoding::$_encodingId",
              &ctx);
        },
        " || ");
    os << ";\n";
  }
  os << R"(    return false;
  };
  if (!isValidPair())
    return ::mlir::failure();
)";
}

/// Emits constraint checks for operands that are NOT overridden by any
/// encoding (i.e. they have a single constraint across all encodings).
static void emitCommonConstraints(const mlir::tblgen::Operator &op,
                                  const llvm::StringSet<> &overriddenNames,
                                  mlir::tblgen::FmtContext &ctx,
                                  raw_ostream &os) {
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    const mlir::tblgen::NamedTypeConstraint &operand = op.getOperand(i);
    if (overriddenNames.contains(operand.name))
      continue;
    std::optional<std::pair<mlir::tblgen::Constraint, bool>> resolved =
        resolveConstraint(operand);
    if (!resolved)
      continue;
    auto [constraint, isOptional] = *resolved;
    emitConstraintCheck(os, ctx, getAdaptorAccessor(operand.name), constraint,
                        isOptional, "::mlir::failure()");
    os << "\n";
  }
}

/// Emits the constraint checks for a single encoding's `checkEnc` lambda body.
/// For each operand, either the encoding provides an override constraint, or
/// the operand's default constraint is used (only for overridden operands).
static void emitEncodingConstraintBody(
    const mlir::tblgen::Operator &op,
    const llvm::StringMap<const llvm::Record *> &overrides,
    const llvm::StringSet<> &overriddenNames, mlir::tblgen::FmtContext &ctx,
    raw_ostream &os) {
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    const mlir::tblgen::NamedTypeConstraint &operand = op.getOperand(i);
    auto it = overrides.find(operand.name);
    if (it != overrides.end()) {
      emitConstraintCheck(os, ctx, getAdaptorAccessor(operand.name),
                          mlir::tblgen::TypeConstraint(it->second),
                          /*isOptional=*/false, "false");
      os << "\n";
      continue;
    }
    if (!overriddenNames.contains(operand.name))
      continue;
    std::optional<std::pair<mlir::tblgen::Constraint, bool>> resolved =
        resolveConstraint(operand);
    if (!resolved)
      continue;
    auto [constraint, isOptional] = *resolved;
    emitConstraintCheck(os, ctx, getAdaptorAccessor(operand.name), constraint,
                        isOptional, "false");
    os << "\n";
  }
}

/// Emits the encoding check block for a single (arch, encoding) pair within
/// an arch-level `if` body.
static void emitSingleEncodingCheck(const mlir::tblgen::Operator &op,
                                    const ArchEncPair &p,
                                    const llvm::StringSet<> &overriddenNames,
                                    mlir::tblgen::FmtContext &ctx,
                                    raw_ostream &os) {
  llvm::StringMap<const llvm::Record *> overrides =
      parseEncodingOverrides(p.enc, op);
  ctx.addSubst("_encodingId", p.encodingId);
  os << mlir::tblgen::tgfmt(
      "    if (encoding == "
      "::mlir::aster::amdgcn::AMDGCNEncoding::$_encodingId) "
      "{\n",
      &ctx);
  os << "      auto checkEnc = [&] {\n";
  emitEncodingConstraintBody(op, overrides, overriddenNames, ctx, os);
  os << R"(        return true;
      };
      if (checkEnc()) return ::mlir::success();
    }
)";
}

/// Emits the per-arch, per-encoding dispatch that checks encoding-specific
/// operand constraints.
static void emitPerEncodingDispatch(const mlir::tblgen::Operator &op,
                                    llvm::ArrayRef<ArchEncPair> pairs,
                                    llvm::ArrayRef<StringRef> archOrder,
                                    const llvm::StringSet<> &overriddenNames,
                                    mlir::tblgen::FmtContext &ctx,
                                    raw_ostream &os) {
  for (StringRef archId : archOrder) {
    ctx.addSubst("_archId", archId);
    os << mlir::tblgen::tgfmt(
        R"(  if (tgt.getTargetFamily() == ::mlir::aster::amdgcn::ISAVersion::$_archId) {
)",
        &ctx);
    for (const ArchEncPair &p : pairs) {
      if (p.archId != archId)
        continue;
      emitSingleEncodingCheck(op, p, overriddenNames, ctx, os);
    }
    os << R"(    return ::mlir::failure();
  }
)";
  }
  os << "  return ::mlir::failure();\n}\n\n";
}

/// Generates the static `isValid<OpName>` function for the given instruction.
static void genIsValidFunc(const InstOp &instOp,
                           llvm::ArrayRef<InstEncRecord> encodings,
                           raw_ostream &os) {
  const mlir::tblgen::Operator &op = instOp.getOperator();
  mlir::tblgen::FmtContext ctx;
  std::string qualClass = instOp.getQualCppClassName();
  std::string funcName = "isValid" + instOp.getCppClassName().str();
  llvm::StringSet<> overriddenNames = collectOverriddenNames(encodings);
  llvm::SmallVector<ArchEncPair> pairs = flattenEncodings(encodings);
  llvm::SmallVector<StringRef> archOrder = getArchOrder(pairs);

  // Function signature.
  ctx.addSubst("_funcName", funcName);
  ctx.addSubst("_qualClass", qualClass);
  const std::string_view sigBody =
      R"(static ::mlir::LogicalResult $_funcName(
    ::mlir::aster::TargetAttrInterface tgt,
    ::mlir::aster::amdgcn::AMDGCNEncoding encoding,
    $_qualClass::Adaptor adaptor) {
)";
  os << mlir::tblgen::tgfmt(sigBody.data(), &ctx);

  emitArchEncodingCheck(pairs, archOrder, ctx, os);
  emitCommonConstraints(op, overriddenNames, ctx, os);
  emitPerEncodingDispatch(op, pairs, archOrder, overriddenNames, ctx, os);
}

//===----------------------------------------------------------------------===//
// Assembly format generation helpers
//===----------------------------------------------------------------------===//

/// Emits a static constexpr C++ array of StringRef arg names.
static void emitNameArray(raw_ostream &os, StringRef indent, StringRef varName,
                          const InstSegmentInfo &seg) {
  if (seg.names.empty()) {
    os << indent << "::llvm::ArrayRef<::llvm::StringRef> " << varName
       << "Names;\n";
    return;
  }
  os << indent << "static constexpr ::llvm::StringRef " << varName
     << "Names[] = {";
  llvm::interleaveComma(seg.names, os,
                        [&](StringRef n) { os << '"' << n << '"'; });
  os << "};\n";
}

/// Emits a static constexpr C++ array of ODSOperandKind values.
static void emitKindArray(raw_ostream &os, StringRef indent, StringRef varName,
                          const InstSegmentInfo &seg) {
  if (seg.kinds.empty()) {
    os << indent << "::llvm::ArrayRef<::mlir::aster::ODSOperandKind> "
       << varName << "Kinds;\n";
    return;
  }
  os << indent << "static constexpr ::mlir::aster::ODSOperandKind " << varName
     << "Kinds[] = {";
  llvm::interleaveComma(seg.kinds, os, [&](ODSOperandKind k) {
    os << getODSOperandKindEnumerator(k);
  });
  os << "};\n";
}

/// Collects the three instruction segments (outs, ins, args).
struct InstSegments {
  InstSegmentInfo outs;
  InstSegmentInfo ins;
  InstSegmentInfo args;
  int numTrailingResults;
};

static InstSegments collectInstSegments(const InstOp &instOp) {
  const mlir::tblgen::Operator &op = instOp.getOperator();
  int numOuts = instOp.getNumOutputs();
  int numIns = instOp.getNumInputs();
  InstSegments segs;
  segs.outs = collectSegmentInfo(op, instOp.getOutputs(), /*odsStart=*/0);
  segs.ins = collectSegmentInfo(op, instOp.getInputs(), /*odsStart=*/numOuts);
  segs.args = collectSegmentInfo(op, instOp.getTrailingArgs(),
                                 /*odsStart=*/numOuts + numIns);
  Dag trailingResults = instOp.getTrailingResults();
  segs.numTrailingResults = static_cast<int>(trailingResults.getNumArgs());
  return segs;
}

/// Returns true if the operator uses the AttrSizedOperandSegments trait.
static bool hasAttrSizedOperandSegments(const mlir::tblgen::Operator &op) {
  return op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments") != nullptr;
}

//===----------------------------------------------------------------------===//
// parse() generation
//===----------------------------------------------------------------------===//

/// Emits the segment size assignment for AttrSizedOperandSegments properties.
static void emitSetSegmentSizes(raw_ostream &os, StringRef className,
                                int totalODSOperands) {
  os << "  // Set operand segment sizes.\n";
  os << "  auto &_props = "
        "result.getOrAddProperties<"
     << className << "::Properties>();\n";
  for (int i = 0; i < totalODSOperands; ++i)
    os << "  _props.operandSegmentSizes[" << i << "] = segmentSizes[" << i
       << "];\n";
}

/// Emits the parse() method body for one instruction.
static void genParseMethod(const InstOp &instOp, StringRef className,
                           const InstSegments &segs, raw_ostream &os) {
  const mlir::tblgen::Operator &op = instOp.getOperator();
  bool hasSegSizes = hasAttrSizedOperandSegments(op);
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_className", className);

  int numOutsODS = static_cast<int>(segs.outs.names.size());
  int numInsODS = static_cast<int>(segs.ins.names.size());
  int numArgsODS = static_cast<int>(segs.args.names.size());
  int totalODS = numOutsODS + numInsODS + numArgsODS;

  // Signature.
  os << mlir::tblgen::tgfmt(
      R"(::mlir::ParseResult
$_className::parse(::mlir::OpAsmParser &parser,
                   ::mlir::OperationState &result) {
)",
      &ctx);

  // Declare operand and type vectors.
  os << "  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand> "
        "outsOperands, insOperands, argsOperands;\n";
  os << "  ::llvm::SmallVector<::mlir::Type> outsTypes, insTypes, "
        "argsTypes;\n";

  // Statically allocate segment sizes in one chunk.
  os << "  std::array<int32_t, " << totalODS << "> segmentSizes{};\n";
  os << "  ::llvm::MutableArrayRef<int32_t> outsSegSizes(segmentSizes.data(), "
     << "static_cast<size_t>(" << numOutsODS << "));\n";
  os << "  ::llvm::MutableArrayRef<int32_t> insSegSizes(segmentSizes.data() + "
     << numOutsODS << ", static_cast<size_t>(" << numInsODS << "));\n";
  os << "  ::llvm::MutableArrayRef<int32_t> argsSegSizes(segmentSizes.data() + "
     << (numOutsODS + numInsODS) << ", static_cast<size_t>(" << numArgsODS
     << "));\n";

  // Emit name/kind arrays for each segment.
  emitNameArray(os, "  ", "outs", segs.outs);
  emitKindArray(os, "  ", "outs", segs.outs);
  emitNameArray(os, "  ", "ins", segs.ins);
  emitKindArray(os, "  ", "ins", segs.ins);
  emitNameArray(os, "  ", "args", segs.args);
  emitKindArray(os, "  ", "args", segs.args);

  // Parse operand segments.
  os << R"(  if (failed(::mlir::aster::parseInstOperands(
          parser, "outs", outsOperands, outsNames, outsKinds, outsSegSizes)))
    return ::mlir::failure();
  if (failed(::mlir::aster::parseInstOperands(
          parser, "ins", insOperands, insNames, insKinds, insSegSizes)))
    return ::mlir::failure();
  if (failed(::mlir::aster::parseInstOperands(
          parser, "args", argsOperands, argsNames, argsKinds, argsSegSizes)))
    return ::mlir::failure();
)";

  // Parse attr-dict.
  os << R"(  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef()
                                     << "' op ";
      })))
      return ::mlir::failure();
  }
)";

  // Parse colon.
  os << "  if (parser.parseColon())\n"
        "    return ::mlir::failure();\n";

  // Parse type segments.
  os << R"(  if (failed(::mlir::aster::parseInstOperandTypes(
          parser, "outs", outsTypes, outsNames, outsKinds, outsSegSizes)))
    return ::mlir::failure();
  if (failed(::mlir::aster::parseInstOperandTypes(
          parser, "ins", insTypes, insNames, insKinds, insSegSizes)))
    return ::mlir::failure();
  if (failed(::mlir::aster::parseInstOperandTypes(
          parser, "args", argsTypes, argsNames, argsKinds, argsSegSizes)))
    return ::mlir::failure();
)";

  // Parse trailing result types.
  if (segs.numTrailingResults > 0) {
    os << R"(  ::llvm::SmallVector<::mlir::Type> trailingResultTypes;
  if (parser.parseArrow() || parser.parseTypeList(trailingResultTypes))
    return ::mlir::failure();
)";
  }

  // Resolve operands.
  os << R"(  // Resolve operands in ODS order: outs, ins, args.
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand> allOperands;
  ::llvm::SmallVector<::mlir::Type> allOperandTypes;
  llvm::append_range(allOperands, outsOperands);
  llvm::append_range(allOperands, insOperands);
  llvm::append_range(allOperands, argsOperands);
  llvm::append_range(allOperandTypes, outsTypes);
  llvm::append_range(allOperandTypes, insTypes);
  llvm::append_range(allOperandTypes, argsTypes);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc,
                             result.operands))
    return ::mlir::failure();
)";

  // Set segment sizes if needed.
  if (hasSegSizes)
    emitSetSegmentSizes(os, className, totalODS);

  // Infer return types.
  os << mlir::tblgen::tgfmt(
      R"(  ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
  if (::mlir::failed($_className::inferReturnTypes(
          parser.getContext(), result.location, result.operands,
          result.attributes.getDictionary(parser.getContext()),
          result.getRawProperties(), result.regions, inferredReturnTypes)))
    return ::mlir::failure();
  result.addTypes(inferredReturnTypes);
)",
      &ctx);

  // Verify parsed trailing result types match inferred ones.
  if (segs.numTrailingResults > 0) {
    os << "  {\n";
    os << "    ::mlir::ArrayRef<::mlir::Type> inferredTrailing =\n";
    os << "        ::mlir::ArrayRef(inferredReturnTypes).take_back("
       << segs.numTrailingResults << ");\n";
    os << "    if (!trailingResultTypes.empty() &&\n";
    os << "        ::mlir::TypeRange(trailingResultTypes) != "
          "::mlir::TypeRange(inferredTrailing))\n";
    os << "      return parser.emitError(parser.getCurrentLocation(),\n";
    os << "          \"trailing result types do not match inferred types\");\n";
    os << "  }\n";
  }

  os << "  return ::mlir::success();\n}\n\n";
}

//===----------------------------------------------------------------------===//
// print() generation
//===----------------------------------------------------------------------===//

/// Emits the list of attribute names to elide from attr-dict printing.
static void emitElidedAttrs(raw_ostream &os, const InstOp &instOp,
                            bool hasSegSizes) {
  const mlir::tblgen::Operator &op = instOp.getOperator();
  os << "  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;\n";
  if (hasSegSizes)
    os << "  elidedAttrs.push_back(\"operandSegmentSizes\");\n";
  if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments"))
    os << "  elidedAttrs.push_back(\"resultSegmentSizes\");\n";
}

/// Emits the print() method body for one instruction.
static void genPrintMethod(const InstOp &instOp, StringRef className,
                           const InstSegments &segs, raw_ostream &os) {
  const mlir::tblgen::Operator &op = instOp.getOperator();
  bool hasSegSizes = hasAttrSizedOperandSegments(op);
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_className", className);

  int numOutsODS = static_cast<int>(segs.outs.names.size());
  int numInsODS = static_cast<int>(segs.ins.names.size());
  int numArgsODS = static_cast<int>(segs.args.names.size());
  int totalODS = numOutsODS + numInsODS + numArgsODS;

  // Signature.
  os << mlir::tblgen::tgfmt(
      R"(void $_className::print(::mlir::OpAsmPrinter &_odsPrinter) {
)",
      &ctx);

  // Allocate segment sizes in one chunk and fill from ODS metadata.
  os << "  std::array<int32_t, " << totalODS << "> segmentSizes{};\n";
  {
    int flatIdx = 0;
    auto emitSegSizeFill = [&](const InstSegmentInfo &seg) {
      for (int odsIdx : seg.odsIndices) {
        os << "  segmentSizes[" << flatIdx
           << "] = std::get<1>(getODSOperandIndexAndLength(" << odsIdx
           << "));\n";
        ++flatIdx;
      }
    };
    emitSegSizeFill(segs.outs);
    emitSegSizeFill(segs.ins);
    emitSegSizeFill(segs.args);
  }
  os << "  ::llvm::ArrayRef<int32_t> outsSegSizes(segmentSizes.data(), "
     << "static_cast<size_t>(" << numOutsODS << "));\n";
  os << "  ::llvm::ArrayRef<int32_t> insSegSizes(segmentSizes.data() + "
     << numOutsODS << ", static_cast<size_t>(" << numInsODS << "));\n";
  os << "  ::llvm::ArrayRef<int32_t> argsSegSizes(segmentSizes.data() + "
     << (numOutsODS + numInsODS) << ", static_cast<size_t>(" << numArgsODS
     << "));\n";

  // Emit operand printing for each segment.
  auto emitPrintOperands = [&](StringRef varName, const InstSegmentInfo &seg) {
    if (seg.names.empty()) {
      os << "  ::mlir::aster::printInstOperands(_odsPrinter, \"" << varName
         << "\", getOperation()->getOperands().slice(0, 0), {}, {}, " << varName
         << "SegSizes);\n";
      return;
    }
    os << "  {\n";
    os << "    auto [_start, _size] = getODSOperandIndexAndLength("
       << seg.odsIndices.front() << ");\n";
    if (seg.odsIndices.size() > 1) {
      int lastIdx = seg.odsIndices.back();
      os << "    auto [_lastStart, _lastSize] = getODSOperandIndexAndLength("
         << lastIdx << ");\n";
      os << "    int64_t _totalSize = (_lastStart + _lastSize) - _start;\n";
    } else {
      os << "    int64_t _totalSize = _size;\n";
    }
    os << "    ::mlir::OperandRange " << varName
       << "Operands = getOperation()->getOperands().slice(_start, "
          "_totalSize);\n";
    emitNameArray(os, "    ", varName, seg);
    emitKindArray(os, "    ", varName, seg);
    os << "    ::mlir::aster::printInstOperands(_odsPrinter, \"" << varName
       << "\", " << varName << "Operands, " << varName << "Names, " << varName
       << "Kinds, " << varName << "SegSizes);\n";
    os << "  }\n";
  };

  emitPrintOperands("outs", segs.outs);
  emitPrintOperands("ins", segs.ins);
  emitPrintOperands("args", segs.args);

  // Print attr-dict.
  emitElidedAttrs(os, instOp, hasSegSizes);
  os << "  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), "
        "elidedAttrs);\n";

  // Print colon and type segments.
  os << "  _odsPrinter << \" :\";\n";

  auto emitPrintTypes = [&](StringRef varName, const InstSegmentInfo &seg) {
    if (seg.names.empty()) {
      os << "  ::mlir::aster::printInstOperandTypes(_odsPrinter, \"" << varName
         << "\", ::mlir::TypeRange(), {}, {}, " << varName << "SegSizes);\n";
      return;
    }
    os << "  {\n";
    os << "    auto [_start, _size] = getODSOperandIndexAndLength("
       << seg.odsIndices.front() << ");\n";
    if (seg.odsIndices.size() > 1) {
      int lastIdx = seg.odsIndices.back();
      os << "    auto [_lastStart, _lastSize] = getODSOperandIndexAndLength("
         << lastIdx << ");\n";
      os << "    int64_t _totalSize = (_lastStart + _lastSize) - _start;\n";
    } else {
      os << "    int64_t _totalSize = _size;\n";
    }
    os << "    ::mlir::OperandRange " << varName
       << "TypeOps = getOperation()->getOperands().slice(_start, "
          "_totalSize);\n";
    emitNameArray(os, "    ", varName, seg);
    emitKindArray(os, "    ", varName, seg);
    os << "    ::mlir::aster::printInstOperandTypes(_odsPrinter, \"" << varName
       << "\", " << varName << "TypeOps.getTypes(), " << varName << "Names, "
       << varName << "Kinds, " << varName << "SegSizes);\n";
    os << "  }\n";
  };

  emitPrintTypes("outs", segs.outs);
  emitPrintTypes("ins", segs.ins);
  emitPrintTypes("args", segs.args);

  // Print trailing result types.
  if (segs.numTrailingResults > 0) {
    os << "  _odsPrinter << \" -> \";\n";
    os << "  llvm::interleaveComma(getOperation()->getResults().take_back("
       << segs.numTrailingResults << ").getTypes(), _odsPrinter);\n";
  }

  os << "}\n\n";
}

//===----------------------------------------------------------------------===//
// Other method generators
//===----------------------------------------------------------------------===//

static void genIsValidMethod(const InstOp &instOp, StringRef className,
                             raw_ostream &os) {
  std::string funcName = "isValid" + instOp.getCppClassName().str();
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_className", className);
  ctx.addSubst("_funcName", funcName);
  const std::string_view body =
      R"(::mlir::LogicalResult
$_className::isValid(::mlir::aster::TargetAttrInterface tgt,
    ::mlir::aster::amdgcn::AMDGCNEncoding encoding,
    $_className::Adaptor adaptor) {
  return $_funcName(tgt, encoding, adaptor);
}

)";
  os << mlir::tblgen::tgfmt(body.data(), &ctx);
}

static void genGetEncoding(const InstOp &instOp, StringRef className,
                           llvm::ArrayRef<InstEncRecord> encodings,
                           raw_ostream &os) {
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_className", className);
  llvm::SmallVector<ArchEncPair> pairs = flattenEncodings(encodings);
  llvm::SmallVector<StringRef> archOrder = getArchOrder(pairs);

  os << "// Instruction: " << instOp.getOperator().getOperationName() << "\n";
  const std::string_view sigBody =
      R"(mlir::FailureOr<::mlir::aster::Encoding>
$_className::getEncoding(::mlir::aster::TargetAttrInterface tgt) {
  Adaptor _adaptor(*this);
)";
  os << mlir::tblgen::tgfmt(sigBody.data(), &ctx);

  for (StringRef archId : archOrder) {
    ctx.addSubst("_archId", archId);
    os << mlir::tblgen::tgfmt(
        R"(  if (tgt.getTargetFamily() == ::mlir::aster::amdgcn::ISAVersion::$_archId) {
)",
        &ctx);
    for (const ArchEncPair &p : pairs) {
      if (p.archId != archId)
        continue;
      ctx.addSubst("_encodingId", p.encodingId);
      os << mlir::tblgen::tgfmt(
          R"(    if (::mlir::succeeded(isValid(tgt, ::mlir::aster::amdgcn::AMDGCNEncoding::$_encodingId, _adaptor)))
      return mlir::aster::Encoding::get(::mlir::aster::amdgcn::AMDGCNEncoding::$_encodingId);
)",
          &ctx);
    }
    os << R"(    return ::mlir::failure();
  }
)";
  }
  os << "  return ::mlir::failure();\n}\n\n";
}

static void genGetEffects(const InstOp &instOp, StringRef className,
                          raw_ostream &os) {
  mlir::tblgen::FmtContext ctx;
  ctx.withSelf("(*this)");
  ctx.addSubst("_className", className);
  ctx.addSubst("_op", "(*this)");
  ctx.addSubst("_effects", "effects");
  const std::string_view sigBody =
      R"(void $_className::getEffects(
    ::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
)";
  os << mlir::tblgen::tgfmt(sigBody.data(), &ctx);
  for (const EffectRecord &effect : instOp.getEffects()) {
    StringRef body = effect.getBody();
    if (body.empty())
      continue;
    os << "  " << mlir::tblgen::tgfmt(body, &ctx) << "\n";
  }
  os << R"(  ::mlir::aster::detail::getInstEffectsImpl(*this, effects);
}

)";
}

static void genInferReturnTypes(const InstOp &instOp, StringRef className,
                                raw_ostream &os) {
  const mlir::tblgen::Operator &op = instOp.getOperator();
  int numOutputs = instOp.getNumOutputs();
  Dag trailingResults = instOp.getTrailingResults();
  int numTrailingResults = static_cast<int>(trailingResults.getNumArgs());

  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_className", className);

  const std::string_view sigBody =
      R"(::llvm::LogicalResult $_className::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::PropertyRef properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  $_className::Adaptor adaptor(operands, attributes, properties, regions);
)";
  os << mlir::tblgen::tgfmt(sigBody.data(), &ctx);

  // Infer result types from output operands.
  for (int i = 0; i < numOutputs; ++i) {
    os << mlir::tblgen::tgfmt(
        R"(  {
    auto [_start, _size] = adaptor.getODSOperandIndexAndLength($0);
    for (::mlir::Type _ty : ::mlir::TypeRange(operands.slice(_start, _size))) {
      auto _regTy = ::llvm::dyn_cast<::mlir::aster::RegisterTypeInterface>(_ty);
      if (_regTy && _regTy.hasValueSemantics())
        inferredReturnTypes.push_back(_ty);
    }
  }
)",
        &ctx, /*0=*/std::to_string(i));
  }

  // Infer buildable trailing result types.
  if (numTrailingResults > 0) {
    int numOutputResults = op.getNumResults() - numTrailingResults;
    mlir::tblgen::FmtContext builderCtx;
    builderCtx.withBuilder("odsBuilder");
    builderCtx.addSubst("_ctxt", "context");

    // Validate that all trailing results are buildable and not optional.
    for (int i = 0; i < numTrailingResults; ++i) {
      const mlir::tblgen::NamedTypeConstraint &result =
          op.getResult(numOutputResults + i);
      if (result.isOptional())
        llvm::PrintFatalError(op.getLoc(), "trailing result '" + result.name +
                                               "' must not be optional");
      if (!result.constraint.getBuilderCall())
        llvm::PrintFatalError(op.getLoc(), "trailing result '" + result.name +
                                               "' must have a buildable type");
    }

    os << "  ::mlir::Builder odsBuilder(context);\n";
    for (int i = 0; i < numTrailingResults; ++i) {
      const mlir::tblgen::NamedTypeConstraint &result =
          op.getResult(numOutputResults + i);
      os << "  inferredReturnTypes.push_back("
         << mlir::tblgen::tgfmt(*result.constraint.getBuilderCall(),
                                &builderCtx)
         << ");\n";
    }
  }

  os << "  return ::mlir::success();\n}\n\n";
}

static void genGetInstInfo(const InstOp &instOp, StringRef className,
                           raw_ostream &os) {
  int numOutputs = instOp.getNumOutputs();
  int numInputs = instOp.getNumInputs();
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_className", className);

  const std::string_view sigBody =
      R"(::mlir::aster::InstOpInfo $_className::getInstInfo() {
  int32_t numInstOuts = 0;
)";
  os << mlir::tblgen::tgfmt(sigBody.data(), &ctx);
  for (int i = 0; i < numOutputs; ++i)
    os << mlir::tblgen::tgfmt(
        "  numInstOuts += std::get<1>(getODSOperandIndexAndLength($0));\n",
        &ctx, /*0=*/std::to_string(i));
  os << "  int32_t numInstIns = 0;\n";
  for (int i = 0; i < numInputs; ++i)
    os << mlir::tblgen::tgfmt(
        "  numInstIns += std::get<1>(getODSOperandIndexAndLength($0));\n", &ctx,
        /*0=*/std::to_string(numOutputs + i));
  os << "  int32_t numInstResults = 0;\n";
  for (int i = 0; i < numOutputs; ++i)
    os << mlir::tblgen::tgfmt(
        "  numInstResults += std::get<1>(getODSResultIndexAndLength($0));\n",
        &ctx, /*0=*/std::to_string(i));
  os << R"(  return ::mlir::aster::InstOpInfo(
      /*numLeadingOperands=*/0, numInstOuts, numInstIns,
      /*numLeadingResults=*/0, numInstResults);
}

)";
}

static void genGetInstProps(const InstOp &instOp, StringRef className,
                            raw_ostream &os) {
  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_className", className);
  llvm::SmallVector<InstProp> props = instOp.getInstProps();
  os << mlir::tblgen::tgfmt(
      R"(const ::mlir::aster::amdgcn::InstructionProps *
$_className::getInstProps() {
  static ::mlir::aster::amdgcn::InstructionProps props({)",
      &ctx);
  llvm::interleaveComma(props, os, [&](const InstProp &prop) {
    os << "::mlir::aster::amdgcn::InstProp::"
       << prop.getAsEnumCase().getIdentifier();
  });
  os << R"(});
  return &props;
}

)";
}

//===----------------------------------------------------------------------===//
// Top-level generators
//===----------------------------------------------------------------------===//

static void genInstMethods(const llvm::Record *rec, raw_ostream &os) {
  InstOp instOp(rec);
  std::string qualClass = instOp.getQualCppClassName();
  StringRef className = StringRef(qualClass).ltrim("::");

  llvm::SmallVector<InstEncRecord> encodings = instOp.getEncodings();
  if (!encodings.empty()) {
    genIsValidFunc(instOp, encodings, os);
    genIsValidMethod(instOp, className, os);
    genGetEncoding(instOp, className, encodings, os);
  }

  genGetEffects(instOp, className, os);
  genInferReturnTypes(instOp, className, os);
  genGetInstInfo(instOp, className, os);
  genGetInstProps(instOp, className, os);

  if (instOp.getGenInstAssemblyFormat()) {
    InstSegments segs = collectInstSegments(instOp);
    genParseMethod(instOp, className, segs, os);
    genPrintMethod(instOp, className, segs, os);
  }
}

static bool generateInstMethods(const llvm::RecordKeeper &records,
                                raw_ostream &os) {
  llvm::IfDefEmitter ifdefEmitter(os, "AMDGCN_GEN_INST_METHODS");
  for (const llvm::Record *rec :
       getSortedDerivedDefinitions(records, AMDISAInstClassType))
    genInstMethods(rec, os);
  return false;
}

//===----------------------------------------------------------------------===//
// TableGen Registration
//===----------------------------------------------------------------------===//

static GenRegistration generateInstMethodsReg(
    "gen-inst-methods", "Generate inst method definitions",
    [](const llvm::RecordKeeper &records, raw_ostream &os) {
      return generateInstMethods(records, os);
    });
