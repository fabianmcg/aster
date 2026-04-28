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
#include "llvm/TableGen/CodeGenHelpers.h"

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
struct EncodingConstraint {
  StringRef argName;
  mlir::tblgen::Constraint constraint;
  bool isOptional;
};

/// Three-state EncType resolution:
///   nullopt           -> incompatible, skip encoding.
///   {nullptr, _}      -> no EncType, no check needed.
///   {rec, isOptional} -> applicable EncType found.
struct EncTypeMatch {
  const llvm::Record *rec;
  bool isOptional;
};

struct EncWithConstraints {
  InstEncRecord enc;
  llvm::SmallVector<EncodingConstraint> constraints;
};

using ByArchMap =
    llvm::MapVector<StringRef, llvm::SmallVector<EncWithConstraints>>;

struct OpEncodingAnalysis {
  llvm::SmallVector<EncodingConstraint> commonConstraints;
  ByArchMap byArch;
};
} // namespace

/// Returns rec if it is an EncType applicable to encArchDef, else null.
static const llvm::Record *matchEncTypeRecord(const llvm::Record *rec,
                                              const llvm::Record *encArchDef) {
  if (!rec->isSubClassOf("EncType"))
    return nullptr;
  for (const llvm::Init *archInit :
       rec->getValueAsListInit("archs")->getElements()) {
    if (cast<llvm::DefInit>(archInit)->getDef() == encArchDef)
      return rec;
  }
  return nullptr;
}

/// Searches rec's allowedTypes for an EncType applicable to encArchDef:
///   nullopt   -> no allowedTypes or none are EncTypes (no check needed).
///   non-null  -> applicable EncType found.
///   null      -> EncTypes present but none apply (skip encoding).
static std::optional<const llvm::Record *>
matchEncTypeViaAllowed(const llvm::Record *rec,
                       const llvm::Record *encArchDef) {
  if (!rec->getValue("allowedTypes"))
    return std::nullopt;
  const llvm::ListInit *allowedTypes = rec->getValueAsListInit("allowedTypes");
  bool hasEncType =
      llvm::any_of(allowedTypes->getElements(), [](const llvm::Init *init) {
        return cast<llvm::DefInit>(init)->getDef()->isSubClassOf("EncType");
      });
  if (!hasEncType)
    return std::nullopt;
  for (const llvm::Init *init : allowedTypes->getElements()) {
    const llvm::Record *typeRec = cast<llvm::DefInit>(init)->getDef();
    if (const llvm::Record *match = matchEncTypeRecord(typeRec, encArchDef))
      return match;
  }
  return static_cast<const llvm::Record *>(nullptr);
}

static std::optional<EncTypeMatch>
findEncTypeForEncoding(const mlir::tblgen::TypeConstraint &tc,
                       const llvm::Record *encArchDef) {
  const llvm::Record *rec = &tc.getDef();
  bool isOptional = false;
  if (tc.isOptional()) {
    rec = rec->getValueAsOptionalDef("baseType");
    if (!rec)
      return EncTypeMatch{nullptr, false};
    isOptional = true;
  }

  if (rec->isSubClassOf("EncType")) {
    if (const llvm::Record *match = matchEncTypeRecord(rec, encArchDef))
      return EncTypeMatch{match, isOptional};
    return std::nullopt;
  }

  std::optional<const llvm::Record *> viaAllowed =
      matchEncTypeViaAllowed(rec, encArchDef);
  if (!viaAllowed)
    return EncTypeMatch{nullptr, false};
  if (!*viaAllowed)
    return std::nullopt;
  return EncTypeMatch{*viaAllowed, isOptional};
}

/// True for operands whose EncType record is the same across all encodings.
static llvm::SmallVector<bool>
computeUniformOperands(const mlir::tblgen::Operator &op,
                       llvm::ArrayRef<InstEncRecord> encodings) {
  int numOps = op.getNumOperands();
  llvm::SmallVector<bool> isUniform(numOps, false);
  for (int i = 0; i < numOps; ++i) {
    const mlir::tblgen::TypeConstraint &tc = op.getOperand(i).constraint;
    const llvm::Record *referenceRec = nullptr;
    bool diverges = false;
    for (const InstEncRecord &enc : encodings) {
      std::optional<EncTypeMatch> match =
          findEncTypeForEncoding(tc, &enc.getEncodedArch().getDef());
      if (!match || !match->rec)
        continue;
      if (!referenceRec) {
        referenceRec = match->rec;
        continue;
      }
      if (match->rec != referenceRec) {
        diverges = true;
        break;
      }
    }
    isUniform[i] = referenceRec && !diverges;
  }
  return isUniform;
}

/// Returns nullopt if any EncType operand is incompatible with this encoding.
/// Operands flagged uniform are skipped (hoisted to common checks).
static std::optional<llvm::SmallVector<EncodingConstraint>>
getConstraintsForEncoding(const mlir::tblgen::Operator &op,
                          const llvm::Record *encArchDef,
                          llvm::ArrayRef<bool> uniformOps) {
  llvm::SmallVector<EncodingConstraint> result;
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    const mlir::tblgen::NamedTypeConstraint &operand = op.getOperand(i);
    std::optional<EncTypeMatch> match =
        findEncTypeForEncoding(operand.constraint, encArchDef);
    if (!match)
      return std::nullopt;
    if (!match->rec || uniformOps[i])
      continue;
    result.push_back({operand.name, mlir::tblgen::TypeConstraint(match->rec),
                      match->isOptional});
  }
  return result;
}

/// Constraints for uniform operands, sampled from the first matching encoding.
static llvm::SmallVector<EncodingConstraint>
computeCommonConstraints(const mlir::tblgen::Operator &op,
                         llvm::ArrayRef<InstEncRecord> encodings,
                         llvm::ArrayRef<bool> uniformOps) {
  llvm::SmallVector<EncodingConstraint> result;
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    if (!uniformOps[i])
      continue;
    const mlir::tblgen::NamedTypeConstraint &operand = op.getOperand(i);
    for (const InstEncRecord &enc : encodings) {
      std::optional<EncTypeMatch> match = findEncTypeForEncoding(
          operand.constraint, &enc.getEncodedArch().getDef());
      if (!match || !match->rec)
        continue;
      result.push_back({operand.name, mlir::tblgen::TypeConstraint(match->rec),
                        match->isOptional});
      break;
    }
  }
  return result;
}

static ByArchMap computeByArch(const mlir::tblgen::Operator &op,
                               llvm::ArrayRef<InstEncRecord> encodings,
                               llvm::ArrayRef<bool> uniformOps) {
  ByArchMap byArch;
  for (const InstEncRecord &enc : encodings) {
    EncodedArchRecord encArch = enc.getEncodedArch();
    const llvm::Record *encArchDef = &encArch.getDef();
    std::optional<llvm::SmallVector<EncodingConstraint>> constraints =
        getConstraintsForEncoding(op, encArchDef, uniformOps);
    if (!constraints)
      continue;
    byArch[encArch.getArch().getIdentifier()].push_back(
        {enc, std::move(*constraints)});
  }
  return byArch;
}

/// Returns nullopt when no encoding survives analysis (nothing to emit).
static std::optional<OpEncodingAnalysis>
analyzeOpEncodings(const mlir::tblgen::Operator &op,
                   llvm::ArrayRef<InstEncRecord> encodings) {
  llvm::SmallVector<bool> uniformOps = computeUniformOperands(op, encodings);
  ByArchMap byArch = computeByArch(op, encodings, uniformOps);
  if (byArch.empty())
    return std::nullopt;
  llvm::SmallVector<EncodingConstraint> commonConstraints =
      computeCommonConstraints(op, encodings, uniformOps);
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
        [&](const EncWithConstraints &ewc) {
          os << llvm::formatv(
              "encoding == ::mlir::aster::amdgcn::Encoding::{0}",
              ewc.enc.getEncodedArch().getEncoding().getIdentifier());
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
static void emitArchDispatch(
    const ByArchMap &byArch, raw_ostream &os,
    llvm::function_ref<void(const EncWithConstraints &)> emitEncoding) {
  for (auto &[archId, encs] : byArch) {
    os << llvm::formatv(
        R"(  if (tgt.getTargetFamily() == ::mlir::aster::TypedEnum::get(::mlir::aster::amdgcn::ISAVersion::{0})) {{
)",
        archId);
    for (const EncWithConstraints &ewc : encs)
      emitEncoding(ewc);
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

  emitArchDispatch(analysis.byArch, os, [&](const EncWithConstraints &ewc) {
    StringRef encodingId =
        ewc.enc.getEncodedArch().getEncoding().getIdentifier();
    os << llvm::formatv(
        "    if (encoding == ::mlir::aster::amdgcn::Encoding::{0}) {{\n",
        encodingId);
    os << "      auto checkEnc = [&] {\n";
    for (const EncodingConstraint &ec : ewc.constraints) {
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

  emitArchDispatch(byArch, os, [&](const EncWithConstraints &ewc) {
    StringRef encodingId =
        ewc.enc.getEncodedArch().getEncoding().getIdentifier();
    os << llvm::formatv(
        R"(    if (::mlir::succeeded(isValid(tgt, ::mlir::aster::amdgcn::Encoding::{0}, _adaptor)))
      return ::mlir::aster::amdgcn::Encoding::{0};
)",
        encodingId);
  });
  os << "  return ::mlir::failure();\n}\n\n";
}

//===----------------------------------------------------------------------===//
// Top-level generators
//===----------------------------------------------------------------------===//

static void genInstMethods(const llvm::Record *rec, raw_ostream &os) {
  mlir::tblgen::Operator op(*rec);
  llvm::SmallVector<InstEncRecord> encodings = getEncodingsFromRecord(*rec);
  if (encodings.empty())
    return;
  std::optional<OpEncodingAnalysis> analysis =
      analyzeOpEncodings(op, encodings);
  if (!analysis)
    return;
  genIsValidOpNameFunc(op, *analysis, os);
  genIsValidMethod(op, os);
  genGetEncoding(op, analysis->byArch, os);
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
