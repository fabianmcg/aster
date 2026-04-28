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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static constexpr StringRef AMDISAInstClassType = "AMDISAInstruction";

static std::string getAdaptorSelf(StringRef name) {
  return buildSelfExpr("adaptor.", "adaptor", name);
}

//===----------------------------------------------------------------------===//
// Encoding constraint analysis
//===----------------------------------------------------------------------===//

namespace {
/// A single per-arg type-constraint check.
struct EncodingConstraint {
  StringRef argName;
  mlir::tblgen::Constraint constraint;
  bool isOptional;
};

/// One (encoding, arch) entry in the per-arch dispatch table. `constraints`
/// holds the checks that must run inside this entry's branch (any op-level
/// constraints already hoisted to the common section are omitted).
struct EncBucket {
  InstEncRecord enc;
  EncodedArchRecord encodedArch;
  llvm::SmallVector<EncodingConstraint> constraints;
};

using ByArchMap = llvm::MapVector<StringRef, llvm::SmallVector<EncBucket>>;

struct OpEncodingAnalysis {
  llvm::SmallVector<EncodingConstraint> commonConstraints;
  ByArchMap byArch;
};

/// Per-encoding override map: argument name -> Type record from the
/// encoding's `constraints` dag.
using OverrideMap = llvm::StringMap<const llvm::Record *>;
} // namespace

/// Build the op-level constraint for `operand`, unwrapping Optional<...>.
/// Returns std::nullopt when an Optional operand has no resolvable baseType.
static std::optional<EncodingConstraint>
makeOpLevelConstraint(const mlir::tblgen::NamedTypeConstraint &operand) {
  const mlir::tblgen::TypeConstraint &tc = operand.constraint;
  const llvm::Record *rec = &tc.getDef();
  bool isOptional = false;
  if (tc.isOptional()) {
    rec = rec->getValueAsOptionalDef("baseType");
    if (!rec)
      return std::nullopt;
    isOptional = true;
  }
  return EncodingConstraint{operand.name, mlir::tblgen::TypeConstraint(rec),
                            isOptional};
}

/// Parse the `constraints` dag of an encoding into a name -> Type record map.
/// Emits a fatal error if a name does not match any operand of `op`.
static OverrideMap parseEncodingOverrides(const InstEncRecord &enc,
                                          const mlir::tblgen::Operator &op) {
  OverrideMap result;
  const llvm::DagInit *dag = enc.getConstraintsDag();
  for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i) {
    StringRef name = dag->getArgNameStr(i);
    const llvm::Init *argInit = dag->getArg(i);
    const auto *defInit = dyn_cast<llvm::DefInit>(argInit);
    if (!defInit) {
      llvm::PrintFatalError(enc.getDef().getLoc(),
                            "encoding constraint argument '" + name +
                                "' must be a Type record");
    }
    bool found = false;
    for (int j = 0, je = op.getNumOperands(); j < je; ++j) {
      if (op.getOperand(j).name == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      llvm::PrintFatalError(
          enc.getDef().getLoc(),
          "encoding constraint references unknown argument '" + name +
              "' in op '" + op.getOperationName() + "'");
    }
    result[name] = defInit->getDef();
  }
  return result;
}

/// Build the per-op encoding analysis. Returns nullopt when there are no
/// encodings to dispatch on.
static std::optional<OpEncodingAnalysis>
analyzeOpEncodings(const mlir::tblgen::Operator &op,
                   llvm::ArrayRef<InstEncRecord> encodings) {
  if (encodings.empty())
    return std::nullopt;

  llvm::SmallVector<OverrideMap> overridesPerEnc;
  overridesPerEnc.reserve(encodings.size());
  llvm::StringSet<> overriddenNames;
  for (const InstEncRecord &enc : encodings) {
    OverrideMap overrides = parseEncodingOverrides(enc, op);
    for (const auto &kv : overrides)
      overriddenNames.insert(kv.getKey());
    overridesPerEnc.push_back(std::move(overrides));
  }

  llvm::SmallVector<EncodingConstraint> commonConstraints;
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    const mlir::tblgen::NamedTypeConstraint &operand = op.getOperand(i);
    if (overriddenNames.contains(operand.name))
      continue;
    std::optional<EncodingConstraint> ec = makeOpLevelConstraint(operand);
    if (!ec)
      continue;
    commonConstraints.push_back(std::move(*ec));
  }

  ByArchMap byArch;
  for (auto [encIdx, enc] : llvm::enumerate(encodings)) {
    const OverrideMap &overrides = overridesPerEnc[encIdx];
    llvm::SmallVector<EncodingConstraint> bucketConstraints;
    for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
      const mlir::tblgen::NamedTypeConstraint &operand = op.getOperand(i);
      auto it = overrides.find(operand.name);
      if (it != overrides.end()) {
        bucketConstraints.push_back({operand.name,
                                     mlir::tblgen::TypeConstraint(it->second),
                                     /*isOptional=*/false});
        continue;
      }
      if (!overriddenNames.contains(operand.name))
        continue;
      std::optional<EncodingConstraint> ec = makeOpLevelConstraint(operand);
      if (!ec)
        continue;
      bucketConstraints.push_back(std::move(*ec));
    }

    for (const EncodedArchRecord &encodedArch : enc.getEncodedArchs()) {
      byArch[encodedArch.getArch().getIdentifier()].push_back(
          {enc, encodedArch, bucketConstraints});
    }
  }

  if (byArch.empty())
    return std::nullopt;
  return OpEncodingAnalysis{std::move(commonConstraints), std::move(byArch)};
}

//===----------------------------------------------------------------------===//
// Code emission
//===----------------------------------------------------------------------===//

/// ctx is taken by value so the caller's context is not mutated. The result
/// is a first-pass string; the caller applies a second tgfmt pass to expand
/// $_self tokens emitted by the condition template.
static std::string
genEncodingConstraint(mlir::tblgen::FmtContext ctx, StringRef self,
                      const mlir::tblgen::Constraint &constraint,
                      bool isOptional, StringRef failureExpr = "false") {
  const std::string_view body = R"(  {
    auto &&_self = $_selfExpr;
    (void)_self;
    if ($0!($1))
      return $2;
  })";
  StrStream stream;
  ctx.addSubst("_selfExpr", self);
  StringRef optionalStr = isOptional ? "_self && " : "";
  stream.os << mlir::tblgen::tgfmt(body.data(), &ctx,
                                   /*0=*/optionalStr,
                                   /*1=*/constraint.getConditionTemplate(),
                                   /*2=*/failureExpr);
  return stream.str;
}

/// Emits an upfront check that returns failure when the (arch, encoding) pair
/// is not covered by byArch.
static void emitEncodingArchPairCheck(const ByArchMap &byArch,
                                      raw_ostream &os) {
  os << "  auto isValidPair = [&]() -> bool {\n";
  for (auto &[archId, encs] : byArch) {
    os << llvm::formatv(
        R"(    if (tgt.getTargetFamily() == ::mlir::aster::TypedEnum::get(::mlir::aster::amdgcn::ISAVersion::{0}))
      return )",
        archId);
    llvm::interleave(
        encs, os,
        [&](const EncBucket &bucket) {
          os << llvm::formatv(
              "encoding == ::mlir::aster::amdgcn::Encoding::{0}",
              bucket.encodedArch.getEncoding().getIdentifier());
        },
        " || ");
    os << ";\n";
  }
  os << "    return false;\n";
  os << "  };\n";
  os << "  if (!isValidPair())\n";
  os << "    return ::mlir::failure();\n";
}

/// Emits the per-arch dispatch skeleton shared by isValid and getEncoding.
static void
emitArchDispatch(const ByArchMap &byArch, raw_ostream &os,
                 llvm::function_ref<void(const EncBucket &)> emitEncoding) {
  for (auto &[archId, encs] : byArch) {
    os << llvm::formatv(
        R"(  if (tgt.getTargetFamily() == ::mlir::aster::TypedEnum::get(::mlir::aster::amdgcn::ISAVersion::{0})) {{
)",
        archId);
    for (const EncBucket &bucket : encs)
      emitEncoding(bucket);
    os << "    return ::mlir::failure();\n";
    os << "  }\n";
  }
}

static void genIsValidOpNameFunc(const mlir::tblgen::Operator &op,
                                 const OpEncodingAnalysis &analysis,
                                 raw_ostream &os) {
  mlir::tblgen::FmtContext ctx;
  ctx.withSelf("_self");

  std::string qualClass = op.getQualCppClassName();
  std::string funcName = "isValid" + op.getCppClassName().str();

  os << llvm::formatv(
      R"(static ::mlir::LogicalResult {0}(
    ::mlir::aster::TargetAttrInterface tgt,
    ::mlir::aster::Encoding encoding,
    {1}::GenericAdaptor<::llvm::ArrayRef<::mlir::aster::ConstValue>> adaptor) {{
)",
      funcName, qualClass);

  emitEncodingArchPairCheck(analysis.byArch, os);

  for (const EncodingConstraint &ec : analysis.commonConstraints) {
    std::string firstPass =
        genEncodingConstraint(ctx, getAdaptorSelf(ec.argName), ec.constraint,
                              ec.isOptional, "::mlir::failure()");
    os << mlir::tblgen::tgfmt(firstPass.data(), &ctx) << "\n";
  }

  emitArchDispatch(analysis.byArch, os, [&](const EncBucket &bucket) {
    StringRef encodingId = bucket.encodedArch.getEncoding().getIdentifier();
    os << llvm::formatv(
        "    if (encoding == ::mlir::aster::amdgcn::Encoding::{0}) {{\n",
        encodingId);
    os << "      auto checkEnc = [&] {\n";
    for (const EncodingConstraint &ec : bucket.constraints) {
      std::string firstPass = genEncodingConstraint(
          ctx, getAdaptorSelf(ec.argName), ec.constraint, ec.isOptional);
      os << mlir::tblgen::tgfmt(firstPass.data(), &ctx) << "\n";
    }
    os << "        return true;\n";
    os << "      };\n";
    os << "      if (checkEnc()) return ::mlir::success();\n";
    os << "    }\n";
  });
  os << "  return ::mlir::failure();\n}\n\n";
}

static void genIsValidMethod(const mlir::tblgen::Operator &op,
                             raw_ostream &os) {
  std::string qualClass = op.getQualCppClassName();
  StringRef className = StringRef(qualClass).ltrim("::");
  std::string funcName = "isValid" + op.getCppClassName().str();

  os << llvm::formatv(
      R"(::mlir::LogicalResult
{0}::isValid(::mlir::aster::TargetAttrInterface tgt,
    ::mlir::aster::Encoding encoding,
    {0}::GenericAdaptor<::llvm::ArrayRef<::mlir::aster::ConstValue>> adaptor) {{
  return {1}(tgt, encoding, adaptor);
}

)",
      className, funcName);
}

static void genGetEncoding(const mlir::tblgen::Operator &op,
                           const ByArchMap &byArch, raw_ostream &os) {
  std::string qualClass = op.getQualCppClassName();
  StringRef className = StringRef(qualClass).ltrim("::");

  os << "// Instruction: " << op.getOperationName() << "\n";
  os << llvm::formatv(
      R"(mlir::FailureOr<::mlir::aster::Encoding>
{0}::getEncoding(::mlir::aster::TargetAttrInterface tgt) {{
  ::llvm::SmallVector<::mlir::aster::ConstValue> _operands;
  _operands.reserve(getNumOperands());
  for (::mlir::Value _v : getOperands())
    _operands.push_back(getTypeOrValue(_v));
  GenericAdaptor<::llvm::ArrayRef<::mlir::aster::ConstValue>> _adaptor(_operands, *this);
)",
      className);

  emitArchDispatch(byArch, os, [&](const EncBucket &bucket) {
    StringRef encodingId = bucket.encodedArch.getEncoding().getIdentifier();
    os << llvm::formatv(
        R"(    if (::mlir::succeeded(isValid(tgt, ::mlir::aster::amdgcn::Encoding::{0}, _adaptor)))
      return ::mlir::aster::amdgcn::Encoding::{0};
)",
        encodingId);
  });
  os << "  return ::mlir::failure();\n}\n\n";
}

static int getNumOutputs(const mlir::tblgen::Operator &op) {
  return op.getDef().getValueAsDag("outputs")->getNumArgs();
}

static int getNumInputs(const mlir::tblgen::Operator &op) {
  return op.getDef().getValueAsDag("inputs")->getNumArgs();
}

/// Emits getEffects: concatenates the body of each effect, then dispatches to
/// the shared impl.
static void genGetEffects(const mlir::tblgen::Operator &op, raw_ostream &os) {
  std::string qualClass = op.getQualCppClassName();
  StringRef className = StringRef(qualClass).ltrim("::");

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
  os << "  ::mlir::aster::detail::getInstEffectsImpl(*this, effects);\n";
  os << "}\n\n";
}

/// Emits inferReturnTypes: appends one type per output operand whose runtime
/// type is a RegisterTypeInterface with value semantics.
static void genInferReturnTypes(const mlir::tblgen::Operator &op,
                                raw_ostream &os) {
  std::string qualClass = op.getQualCppClassName();
  StringRef className = StringRef(qualClass).ltrim("::");
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

/// Emits getInstInfo, computing operand and result segment counts via
/// getODSOperandIndexAndLength and getODSResultIndexAndLength.
static void genGetInstInfo(const mlir::tblgen::Operator &op, raw_ostream &os) {
  std::string qualClass = op.getQualCppClassName();
  StringRef className = StringRef(qualClass).ltrim("::");
  int numOutputs = getNumOutputs(op);
  int numInputs = getNumInputs(op);

  os << llvm::formatv("::mlir::aster::InstOpInfo {0}::getInstInfo() {{\n"
                      "  int32_t numInstOuts = 0;\n",
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
  os << "  return ::mlir::aster::InstOpInfo(\n"
        "      /*numLeadingOperands=*/0, numInstOuts, numInstIns,\n"
        "      /*numLeadingResults=*/0, numInstResults);\n"
        "}\n\n";
}

//===----------------------------------------------------------------------===//
// Top-level generators
//===----------------------------------------------------------------------===//

static void genInstMethods(const llvm::Record *rec, raw_ostream &os) {
  mlir::tblgen::Operator op(*rec);

  llvm::SmallVector<InstEncRecord> encodings = getEncodingsFromRecord(*rec);
  std::optional<OpEncodingAnalysis> analysis;
  if (!encodings.empty())
    analysis = analyzeOpEncodings(op, encodings);
  if (analysis) {
    genIsValidOpNameFunc(op, *analysis, os);
    genIsValidMethod(op, os);
    genGetEncoding(op, analysis->byArch, os);
  }

  genGetEffects(op, os);
  genInferReturnTypes(op, os);
  genGetInstInfo(op, os);
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
