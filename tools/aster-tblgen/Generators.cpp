//===- Generators.cpp - Generator registrations for mlir-tblgen -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file registers all generators for mlir-tblgen by calling into the
// MLIRTableGenCppGen library. CLI options are read here and threaded as
// explicit parameters to the library functions.
//
//===----------------------------------------------------------------------===//

#include "OpGen.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Generators/AttrOrTypeDefGen.h"
#include "mlir/TableGen/Generators/BytecodeDialectGen.h"
#include "mlir/TableGen/Generators/DialectGen.h"
#include "mlir/TableGen/Generators/DialectInterfacesGen.h"
#include "mlir/TableGen/Generators/EnumPythonBindingGen.h"
#include "mlir/TableGen/Generators/EnumsGen.h"
#include "mlir/TableGen/Generators/FormatGen.h"
#include "mlir/TableGen/Generators/OpDefinitionsGen.h"
#include "mlir/TableGen/Generators/OpDocGen.h"
#include "mlir/TableGen/Generators/OpGenHelpers.h"
#include "mlir/TableGen/Generators/OpInterfacesGen.h"
#include "mlir/TableGen/Generators/OpPythonBindingGen.h"
#include "mlir/TableGen/Generators/PassCAPIGen.h"
#include "mlir/TableGen/Generators/PassDocGen.h"
#include "mlir/TableGen/Generators/PassGen.h"
#include "mlir/TableGen/Generators/RewriterGen.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Assembly format options (shared by AttrOrTypeDef and Op generators)
//===----------------------------------------------------------------------===//

static cl::opt<bool>
    formatErrorIsFatal("asmformat-error-is-fatal",
                       cl::desc("Emit a fatal error if format parsing fails"),
                       cl::init(true));

//===----------------------------------------------------------------------===//
// Op definition generator options (shared by op-def and op-doc generators)
//===----------------------------------------------------------------------===//

static cl::OptionCategory opDefGenCat("Options for op definition generators");
static cl::opt<std::string> opIncFilter(
    "op-include-regex",
    cl::desc("Regex of name of op's to include (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<std::string> opExcFilter(
    "op-exclude-regex",
    cl::desc("Regex of name of op's to exclude (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<unsigned> opShardCount(
    "op-shard-count",
    cl::desc("The number of shards into which the op classes will be divided"),
    cl::cat(opDefGenCat), cl::init(1));

static std::vector<const Record *>
getRequestedOpDefs(const RecordKeeper &records) {
  return getRequestedOpDefinitions(records, opIncFilter, opExcFilter);
}

static void shardOps(ArrayRef<const Record *> defs,
                     SmallVectorImpl<ArrayRef<const Record *>> &shardedDefs) {
  shardOpDefinitions(defs, shardedDefs, opShardCount);
}

//===----------------------------------------------------------------------===//
// Enum generators
//===----------------------------------------------------------------------===//

static GenRegistration
    genEnumDecls("gen-enum-decls", "Generate enum utility declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return emitEnumDecls(records, os);
                 });
static GenRegistration
    genEnumDefs("gen-enum-defs", "Generate enum utility definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitEnumDefs(records, os);
                });

//===----------------------------------------------------------------------===//
// Op definition generators
//===----------------------------------------------------------------------===//

static GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 std::vector<const Record *> defs = getRequestedOpDefs(records);
                 SmallVector<ArrayRef<const Record *>> shardedDefs;
                 shardOps(defs, shardedDefs);
                 return emitOpDecls(records, defs, shardedDefs.size(), os,
                                    formatErrorIsFatal);
               });
static GenRegistration
    genOpDefs("gen-op-defs", "Generate op definitions",
              [](const RecordKeeper &records, raw_ostream &os) {
                std::vector<const Record *> defs = getRequestedOpDefs(records);
                SmallVector<ArrayRef<const Record *>> shardedDefs;
                shardOps(defs, shardedDefs);
                return emitOpDefs(records, defs, shardedDefs.size(), os,
                                  formatErrorIsFatal);
              });

//===----------------------------------------------------------------------===//
// Aster op definition generators
//===----------------------------------------------------------------------===//

static GenRegistration genAsterOpDecls(
    "gen-aster-op-decls", "Generate Aster op declarations",
    [](const RecordKeeper &records, raw_ostream &os) {
      std::vector<const Record *> defs = getRequestedOpDefs(records);
      SmallVector<ArrayRef<const Record *>> shardedDefs;
      shardOps(defs, shardedDefs);
      return aster::emitOpDecls(records, defs, shardedDefs.size(), os,
                                formatErrorIsFatal);
    });
static GenRegistration
    genAsterOpDefs("gen-aster-op-defs", "Generate Aster op definitions",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     std::vector<const Record *> defs =
                         getRequestedOpDefs(records);
                     SmallVector<ArrayRef<const Record *>> shardedDefs;
                     shardOps(defs, shardedDefs);
                     return aster::emitOpDefs(records, defs, shardedDefs.size(),
                                              os, formatErrorIsFatal);
                   });
