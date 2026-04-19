//===- OpGen.cpp - Aster op definitions generator -------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Aster op definitions generator. It introduces
// AsterOpOperandAdaptorEmitter and AsterOpEmitter as extension points for
// customising the generated C++ op class declarations and definitions.
//
//===----------------------------------------------------------------------===//

#include "OpGen.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Generators/CppGenUtilities.h"
#include "mlir/TableGen/Generators/OpAdaptorHelper.h"
#include "mlir/TableGen/Generators/OpDefinitionsGen.h"
#include "mlir/TableGen/Generators/OpGenHelpers.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

/// {0}: Op or adaptor name.
/// {1}: "declarations" or "definitions".
static const char *const opCommentHeader = R"(
//===----------------------------------------------------------------------===//
// {0} {1}
//===----------------------------------------------------------------------===//

)";

//===----------------------------------------------------------------------===//
// AsterOpOperandAdaptorEmitter
//===----------------------------------------------------------------------===//

/// Generates C++ declarations and definitions for the operand adaptor of a
/// single operation. Mirrors the interface of the MLIR-internal
/// OpOperandAdaptorEmitter and serves as the customisation point for
/// Aster-specific adaptor generation.
class AsterOpOperandAdaptorEmitter {
public:
  static void emitDecl(const Operator &op,
                       const StaticVerifierFunctionEmitter &sve,
                       raw_ostream &os);
  static void emitDef(const Operator &op,
                      const StaticVerifierFunctionEmitter &sve,
                      raw_ostream &os);

private:
  explicit AsterOpOperandAdaptorEmitter(
      const Operator &op, const StaticVerifierFunctionEmitter &sve);

  const Operator &op;
  const StaticVerifierFunctionEmitter &staticVerifierEmitter;
};

AsterOpOperandAdaptorEmitter::AsterOpOperandAdaptorEmitter(
    const Operator &op, const StaticVerifierFunctionEmitter &sve)
    : op(op), staticVerifierEmitter(sve) {}

void AsterOpOperandAdaptorEmitter::emitDecl(
    const Operator &op, const StaticVerifierFunctionEmitter &sve,
    raw_ostream &os) {
  // TODO: Implement Aster-specific adaptor declarations.
}

void AsterOpOperandAdaptorEmitter::emitDef(
    const Operator &op, const StaticVerifierFunctionEmitter &sve,
    raw_ostream &os) {
  // TODO: Implement Aster-specific adaptor definitions.
}

//===----------------------------------------------------------------------===//
// AsterOpEmitter
//===----------------------------------------------------------------------===//

/// Generates C++ declarations and definitions for a single Aster operation.
/// Subclasses mlir::tblgen::OpEmitter to inherit the full upstream generation
/// logic; individual virtual gen* hooks can be overridden to customise output.
class AsterOpEmitter : public mlir::tblgen::OpEmitter {
public:
  static void emitDecl(const Operator &op, raw_ostream &os,
                       const StaticVerifierFunctionEmitter &sve,
                       bool fatalOnError = true);
  static void emitDef(const Operator &op, raw_ostream &os,
                      const StaticVerifierFunctionEmitter &sve,
                      bool fatalOnError = true);

protected:
  AsterOpEmitter(const Operator &op, const StaticVerifierFunctionEmitter &sve,
                 bool fatalOnError = true);
};

AsterOpEmitter::AsterOpEmitter(const Operator &op,
                               const StaticVerifierFunctionEmitter &sve,
                               bool fatalOnError)
    : OpEmitter(op, sve, fatalOnError) {}

void AsterOpEmitter::emitDecl(const Operator &op, raw_ostream &os,
                              const StaticVerifierFunctionEmitter &sve,
                              bool fatalOnError) {
  AsterOpEmitter emitter(op, sve, fatalOnError);
  emitter.OpEmitter::emitDecl(os);
}

void AsterOpEmitter::emitDef(const Operator &op, raw_ostream &os,
                             const StaticVerifierFunctionEmitter &sve,
                             bool fatalOnError) {
  AsterOpEmitter emitter(op, sve, fatalOnError);
  emitter.OpEmitter::emitDef(os);
}

//===----------------------------------------------------------------------===//
// Emission infrastructure
//===----------------------------------------------------------------------===//

/// Emit declarations or definitions for each op in \p defs using the Aster
/// emitters.
static void emitOpClasses(const RecordKeeper &records,
                          ArrayRef<const Record *> defs, raw_ostream &os,
                          const StaticVerifierFunctionEmitter &sve,
                          bool emitDecl, bool fatalOnError = true) {
  if (defs.empty())
    return;

  for (const Record *def : defs) {
    Operator op(*def);
    OpOrAdaptorHelper emitHelper(op, /*emitForOp=*/true);
    if (emitDecl) {
      {
        NamespaceEmitter emitter(os, op.getCppNamespace());
        os << formatv(opCommentHeader, op.getQualCppClassName(),
                      "declarations");
        AsterOpOperandAdaptorEmitter::emitDecl(op, sve, os);
        AsterOpEmitter::emitDecl(op, os, sve, fatalOnError);
      }
      if (!op.getCppNamespace().empty()) {
        os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
           << "::" << op.getCppClassName() << ")\n";
        if (emitHelper.hasNonEmptyPropertiesStruct())
          os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
             << "::detail::" << op.getCppClassName()
             << "GenericAdaptorBase::Properties)\n";
        os << "\n";
      }
    } else {
      {
        NamespaceEmitter emitter(os, op.getCppNamespace());
        os << formatv(opCommentHeader, op.getQualCppClassName(), "definitions");
        AsterOpOperandAdaptorEmitter::emitDef(op, sve, os);
        AsterOpEmitter::emitDef(op, os, sve, fatalOnError);
      }
      if (!op.getCppNamespace().empty()) {
        os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
           << "::" << op.getCppClassName() << ")\n";
        if (emitHelper.hasNonEmptyPropertiesStruct())
          os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
             << "::detail::" << op.getCppClassName()
             << "GenericAdaptorBase::Properties)\n";
        os << "\n";
      }
    }
  }
}

/// Emit forward declarations and class declarations for all ops in \p defs.
static void emitOpClassDecls(const RecordKeeper &records,
                             ArrayRef<const Record *> defs, raw_ostream &os,
                             bool fatalOnError = true) {
  for (const Record *def : defs) {
    Operator op(*def);
    NamespaceEmitter emitter(os, op.getCppNamespace());
    tblgen::emitSummaryAndDescComments(os, op.getSummary(),
                                       op.getDescription());
    os << "class " << op.getCppClassName() << ";\n";
  }

  IfDefEmitter scope(os, "GET_OP_CLASSES");
  if (defs.empty())
    return;
  StaticVerifierFunctionEmitter staticVerifierEmitter(os, records);
  staticVerifierEmitter.collectOpConstraints(defs);
  emitOpClasses(records, defs, os, staticVerifierEmitter,
                /*emitDecl=*/true, fatalOnError);
}

/// Emit op class definitions for all ops in \p defs.
static void emitOpClassDefs(const RecordKeeper &records,
                            ArrayRef<const Record *> defs, raw_ostream &os,
                            bool fatalOnError = true) {
  if (defs.empty())
    return;

  StaticVerifierFunctionEmitter staticVerifierEmitter(os, records);
  os << formatv(opCommentHeader, "Local Utility Method", "Definitions");
  staticVerifierEmitter.collectOpConstraints(defs);
  staticVerifierEmitter.emitOpConstraints();

  emitOpClasses(records, defs, os, staticVerifierEmitter,
                /*emitDecl=*/false, fatalOnError);
}

//===----------------------------------------------------------------------===//
// Top-level entry points
//===----------------------------------------------------------------------===//

bool aster::emitOpDecls(const RecordKeeper &records,
                        ArrayRef<const Record *> defs, unsigned shardCount,
                        raw_ostream &os, bool fatalOnError) {
  emitSourceFileHeader("Op Declarations", os, records);
  emitOpClassDecls(records, defs, os, fatalOnError);
  return false;
}

bool aster::emitOpDefs(const RecordKeeper &records,
                       ArrayRef<const Record *> defs, unsigned shardCount,
                       raw_ostream &os, bool fatalOnError) {
  emitSourceFileHeader("Op Definitions", os, records);

  {
    IfDefEmitter scope(os, "GET_OP_LIST");
    interleave(
        defs, os,
        [&](const Record *def) { os << Operator(def).getQualCppClassName(); },
        ",\n");
  }
  {
    IfDefEmitter scope(os, "GET_OP_CLASSES");
    emitOpClassDefs(records, defs, os, fatalOnError);
  }
  return false;
}
