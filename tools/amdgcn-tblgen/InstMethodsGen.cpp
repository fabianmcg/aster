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
    auto &&_self = getTypeOrValue($0;);
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
collectOverriddenNames(const mlir::tblgen::Operator &op,
                       llvm::ArrayRef<InstEncRecord> encodings) {
  llvm::StringSet<> overriddenNames;
  for (const InstEncRecord &enc : encodings) {
    const llvm::DagInit *dag = enc.getConstraintsDag();
    for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i)
      overriddenNames.insert(dag->getArgNameStr(i));
  }
  return overriddenNames;
}

static int getNumOutputs(const mlir::tblgen::Operator &op) {
  return op.getDef().getValueAsDag("outputs")->getNumArgs();
}

static int getNumInputs(const mlir::tblgen::Operator &op) {
  return op.getDef().getValueAsDag("inputs")->getNumArgs();
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
// Code emission
//===----------------------------------------------------------------------===//

static void genIsValidFunc(const mlir::tblgen::Operator &op,
                           StringRef className,
                           llvm::ArrayRef<InstEncRecord> encodings,
                           raw_ostream &os) {
  mlir::tblgen::FmtContext ctx;
  std::string qualClass = op.getQualCppClassName();
  std::string funcName = "isValid" + op.getCppClassName().str();
  llvm::StringSet<> overriddenNames = collectOverriddenNames(op, encodings);
  llvm::SmallVector<ArchEncPair> pairs = flattenEncodings(encodings);
  llvm::SmallVector<StringRef> archOrder = getArchOrder(pairs);

  // Function signature.
  os << llvm::formatv(
      R"(static ::mlir::LogicalResult {0}(
    ::mlir::aster::TargetAttrInterface tgt,
    ::mlir::aster::Encoding encoding,
    {1}::Adaptor adaptor) {{
)",
      funcName, qualClass);

  // Emit the (arch, encoding) pair validity check.
  os << "  auto isValidPair = [&]() -> bool {\n";
  for (StringRef archId : archOrder) {
    os << llvm::formatv(
        R"(    if (tgt.getTargetFamily() == ::mlir::aster::TypedEnum::get(::mlir::aster::amdgcn::ISAVersion::{0}))
      return )",
        archId);
    llvm::interleave(
        llvm::make_filter_range(
            pairs, [&](const ArchEncPair &p) { return p.archId == archId; }),
        os,
        [&](const ArchEncPair &p) {
          os << llvm::formatv(
              "encoding == ::mlir::aster::amdgcn::Encoding::{0}", p.encodingId);
        },
        " || ");
    os << ";\n";
  }
  os << R"(    return false;
  };
  if (!isValidPair())
    return ::mlir::failure();
)";

  // Emit common (non-overridden) constraints.
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

  // Emit per-arch, per-encoding dispatch.
  for (StringRef archId : archOrder) {
    os << llvm::formatv(
        R"(  if (tgt.getTargetFamily() == ::mlir::aster::TypedEnum::get(::mlir::aster::amdgcn::ISAVersion::{0})) {{
)",
        archId);
    for (const ArchEncPair &p : pairs) {
      if (p.archId != archId)
        continue;
      llvm::StringMap<const llvm::Record *> overrides =
          parseEncodingOverrides(p.enc, op);
      os << llvm::formatv(
          "    if (encoding == ::mlir::aster::amdgcn::Encoding::{0}) {{\n",
          p.encodingId);
      os << "      auto checkEnc = [&] {\n";
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
        emitConstraintCheck(os, ctx, getAdaptorAccessor(operand.name),
                            constraint, isOptional, "false");
        os << "\n";
      }
      os << R"(        return true;
      };
      if (checkEnc()) return ::mlir::success();
    }
)";
    }
    os << R"(    return ::mlir::failure();
  }
)";
  }
  os << "  return ::mlir::failure();\n}\n\n";
}

static void genIsValidMethod(const mlir::tblgen::Operator &op,
                             StringRef className, raw_ostream &os) {
  std::string funcName = "isValid" + op.getCppClassName().str();
  os << llvm::formatv(
      R"(::mlir::LogicalResult
{0}::isValid(::mlir::aster::TargetAttrInterface tgt,
    ::mlir::aster::Encoding encoding,
    {0}::Adaptor adaptor) {{
  return {1}(tgt, encoding, adaptor);
}

)",
      className, funcName);
}

static void genGetEncoding(const mlir::tblgen::Operator &op,
                           StringRef className,
                           llvm::ArrayRef<InstEncRecord> encodings,
                           raw_ostream &os) {
  llvm::SmallVector<ArchEncPair> pairs = flattenEncodings(encodings);
  llvm::SmallVector<StringRef> archOrder = getArchOrder(pairs);

  os << "// Instruction: " << op.getOperationName() << "\n";
  os << llvm::formatv(
      R"(mlir::FailureOr<::mlir::aster::Encoding>
{0}::getEncoding(::mlir::aster::TargetAttrInterface tgt) {{
  Adaptor _adaptor(*this);
)",
      className);

  for (StringRef archId : archOrder) {
    os << llvm::formatv(
        R"(  if (tgt.getTargetFamily() == ::mlir::aster::TypedEnum::get(::mlir::aster::amdgcn::ISAVersion::{0})) {{
)",
        archId);
    for (const ArchEncPair &p : pairs) {
      if (p.archId != archId)
        continue;
      os << llvm::formatv(
          R"(    if (::mlir::succeeded(isValid(tgt, ::mlir::aster::amdgcn::Encoding::{0}, _adaptor)))
      return ::mlir::aster::amdgcn::Encoding::{0};
)",
          p.encodingId);
    }
    os << R"(    return ::mlir::failure();
  }
)";
  }
  os << "  return ::mlir::failure();\n}\n\n";
}

static void genGetEffects(const mlir::tblgen::Operator &op, StringRef className,
                          raw_ostream &os) {
  os << llvm::formatv(
      R"(void {0}::getEffects(
    ::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {{
)",
      className);

  mlir::tblgen::FmtContext ctx;
  ctx.addSubst("_op", "(*this)");
  ctx.addSubst("_effects", "effects");
  const llvm::ListInit *effectsList = op.getDef().getValueAsListInit("effects");
  for (const llvm::Init *init : effectsList->getElements()) {
    const llvm::Record *effectRec = cast<llvm::DefInit>(init)->getDef();
    StringRef body = effectRec->getValueAsString("body");
    if (body.empty())
      continue;
    os << "  " << mlir::tblgen::tgfmt(body, &ctx) << "\n";
  }
  os << R"(  ::mlir::aster::detail::getInstEffectsImpl(*this, effects);
}

)";
}

static void genInferReturnTypes(const mlir::tblgen::Operator &op,
                                StringRef className, raw_ostream &os) {
  int numOutputs = getNumOutputs(op);

  os << llvm::formatv(
      R"(::llvm::LogicalResult {0}::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::PropertyRef properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {{
  {0}::Adaptor adaptor(operands, attributes, properties, regions);
)",
      className);

  for (int i = 0; i < numOutputs; ++i) {
    os << llvm::formatv(
        R"(  {{
    auto [_start, _size] = adaptor.getODSOperandIndexAndLength({0});
    for (::mlir::Type _ty : ::mlir::TypeRange(operands.slice(_start, _size))) {{
      auto _regTy = ::llvm::dyn_cast<::mlir::aster::RegisterTypeInterface>(_ty);
      if (_regTy && _regTy.hasValueSemantics())
        inferredReturnTypes.push_back(_ty);
    }
  }
)",
        i);
  }
  os << "  return ::mlir::success();\n}\n\n";
}

static void genGetInstInfo(const mlir::tblgen::Operator &op,
                           StringRef className, raw_ostream &os) {
  int numOutputs = getNumOutputs(op);
  int numInputs = getNumInputs(op);

  os << llvm::formatv(
      R"(::mlir::aster::InstOpInfo {0}::getInstInfo() {{
  int32_t numInstOuts = 0;
)",
      className);
  for (int i = 0; i < numOutputs; ++i)
    os << llvm::formatv(
        "  numInstOuts += std::get<1>(getODSOperandIndexAndLength({0}));\n", i);
  os << "  int32_t numInstIns = 0;\n";
  for (int i = 0; i < numInputs; ++i)
    os << llvm::formatv(
        "  numInstIns += std::get<1>(getODSOperandIndexAndLength({0}));\n",
        numOutputs + i);
  os << "  int32_t numInstResults = 0;\n";
  for (int i = 0; i < numOutputs; ++i)
    os << llvm::formatv(
        "  numInstResults += std::get<1>(getODSResultIndexAndLength({0}));\n",
        i);
  os << R"(  return ::mlir::aster::InstOpInfo(
      /*numLeadingOperands=*/0, numInstOuts, numInstIns,
      /*numLeadingResults=*/0, numInstResults);
}

)";
}

//===----------------------------------------------------------------------===//
// Top-level generators
//===----------------------------------------------------------------------===//

static void genInstMethods(const llvm::Record *rec, raw_ostream &os) {
  mlir::tblgen::Operator op(*rec);
  std::string qualClass = op.getQualCppClassName();
  StringRef className = StringRef(qualClass).ltrim("::");

  llvm::SmallVector<InstEncRecord> encodings = getEncodingsFromRecord(*rec);
  if (!encodings.empty()) {
    genIsValidFunc(op, className, encodings, os);
    genIsValidMethod(op, className, os);
    genGetEncoding(op, className, encodings, os);
  }

  genGetEffects(op, className, os);
  genInferReturnTypes(op, className, os);
  genGetInstInfo(op, className, os);
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
