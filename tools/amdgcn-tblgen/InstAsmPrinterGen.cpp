//===- InstAsmPrinterGen.cpp ----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates encoding-aware assembly printers for AMDGCN instructions.
//
//===----------------------------------------------------------------------===//

#include "InstCommon.h"
#include "aster/Support/Lexer.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/Error.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::tblgen;

//===----------------------------------------------------------------------===//
// Generate encoding-aware asm printers for AMDISAInstruction records.
//===----------------------------------------------------------------------===//

namespace {
/// Handler to generate the encoding-aware asm printer for an instruction.
struct ASMPrinterHandler {
  ASMPrinterHandler(const llvm::Record *rec);
  void genPrinter(raw_ostream &os);

private:
  using ArgTy = std::optional<std::pair<DagArg, ASMArgFormat>>;
  /// Emit the printer code for a single ASMString syntax.
  void emitSyntax(StringRef syntax, mlir::raw_indented_ostream &os);
  /// Emit the printer code for a single argument reference.
  void emitArg(DagArg dagArg, ASMArgFormat arg, mlir::raw_indented_ostream &os);
  /// Emit an error message.
  void emitError(Twine msg) { llvm::PrintFatalError(rec, msg); }
  const llvm::Record *rec;
  mlir::tblgen::Operator op;
  mlir::tblgen::FmtContext ctx;
  llvm::StringMap<ArgTy> arguments;
  StringRef mnemonic;
};
} // namespace

ASMPrinterHandler::ASMPrinterHandler(const llvm::Record *rec)
    : rec(rec), op(*rec) {
  mnemonic = rec->getValueAsString("opName");
  // Collect arguments from outputs, inputs, and trailingArgs.
  for (StringRef dagField : {"outputs", "inputs", "trailingArgs"}) {
    Dag dag(rec->getValueAsDag(dagField));
    for (auto [i, arg] : llvm::enumerate(dag.getAsRange())) {
      if (!ASMArgFormat::isa(arg.getAsRecord()))
        continue;
      arguments[arg.getName()] = {arg, ASMArgFormat(arg.getAsRecord())};
    }
  }
  // Set up the format context.
  ctx.addSubst("_inst", "_inst");
  ctx.addSubst("_printer", "printer");
}

void ASMPrinterHandler::emitArg(DagArg dagArg, ASMArgFormat arg,
                                mlir::raw_indented_ostream &os) {
  ctx.withSelf("_inst.get" +
               llvm::convertToCamelFromSnakeCase(dagArg.getName(), true) +
               "()");
  os.printReindented(mlir::tblgen::tgfmt(arg.getPrinter(), &ctx).str());
  os << "\n";
  ctx.withSelf("_inst");
}

void ASMPrinterHandler::emitSyntax(StringRef syntax,
                                   mlir::raw_indented_ostream &os) {
  Lexer lexer(syntax);
  while (lexer.currentChar() != '\0') {
    lexer.consumeWhiteSpace();
    if (lexer.currentChar() == '\0')
      break;

    // Handle `${mnemonic}` interpolation.
    if (lexer.currentChar() == '$' && lexer.getCurrentPos().size() > 1 &&
        lexer.getCurrentPos()[1] == '{') {
      lexer.consumeChar(); // consume '$'
      lexer.consumeChar(); // consume '{'
      FailureOr<StringRef> id = lexer.lexIdentifier();
      if (failed(id))
        emitError("failed to lex interpolation in asm_syntax");
      if (*id != "mnemonic")
        emitError("unknown interpolation ${" + *id + "} in asm_syntax");
      if (lexer.currentChar() != '}')
        emitError("expected '}' in asm_syntax interpolation");
      lexer.consumeChar(); // consume '}'

      // Collect any trailing suffix (e.g. `_e64` in `${mnemonic}_e64`).
      std::string suffix;
      while (lexer.currentChar() == '_' || std::isalnum(lexer.currentChar())) {
        suffix += lexer.currentChar();
        lexer.consumeChar();
      }
      os << "$_printer.printMnemonic(\"" << mnemonic << suffix << "\");\n";
      continue;
    }

    // Handle `$identifier` -- operand/input reference.
    if (lexer.currentChar() == '$') {
      lexer.consumeChar();
      FailureOr<StringRef> id = lexer.lexIdentifier();
      if (failed(id))
        emitError("failed to lex identifier in asm_syntax: " +
                  lexer.getCurrentPos());

      ArgTy arg = arguments.lookup(*id);
      if (!arg.has_value())
        emitError("unknown operand $" + *id + " in asm_syntax");

      emitArg(arg->first, arg->second, os);
      continue;
    }

    // Handle `[identifier]` -- modifier reference.
    if (lexer.currentChar() == '[') {
      lexer.consumeChar(); // consume '['
      lexer.consumeWhiteSpace();
      FailureOr<StringRef> id = lexer.lexIdentifier();
      if (failed(id))
        emitError("failed to lex modifier name in asm_syntax: " +
                  lexer.getCurrentPos());
      lexer.consumeWhiteSpace();
      if (lexer.currentChar() != ']')
        emitError("expected ']' after modifier name in asm_syntax");
      lexer.consumeChar(); // consume ']'

      ArgTy arg = arguments.lookup(*id);
      if (!arg.has_value())
        emitError("unknown modifier [" + *id + "] in asm_syntax");

      emitArg(arg->first, arg->second, os);
      continue;
    }

    // Handle comma.
    if (lexer.currentChar() == ',') {
      lexer.consumeChar();
      os << "$_printer.printComma();\n";
      continue;
    }

    // Handle keywords.
    if (lexer.currentChar() == '_' || std::isalpha(lexer.currentChar())) {
      FailureOr<StringRef> id = lexer.lexIdentifier();
      if (failed(id))
        emitError("failed to lex keyword in asm_syntax: " +
                  lexer.getCurrentPos());
      os << llvm::formatv("$_printer.printKeyword(\"{0}\");\n", *id);
      continue;
    }

    // Unexpected character.
    emitError("unexpected character in asm_syntax: " + lexer.getCurrentPos());
  }
  os << "return success();\n";
}

void ASMPrinterHandler::genPrinter(raw_ostream &osOut) {
  StrStream strStream;
  mlir::raw_indented_ostream os(strStream.os);
  std::string qualClass = op.getQualCppClassName();

  // Read the asm_syntax list.
  llvm::SmallVector<ASMStringRecord> asmStrings = llvm::map_to_vector(
      rec->getValueAsListInit("asm_syntax")->getElements(),
      [](const llvm::Init *init) {
        return ASMStringRecord(cast<llvm::DefInit>(init)->getDef());
      });
  if (asmStrings.empty())
    return;

  // Build a flat list of (arch, encoding, syntax, predicate) entries.
  struct ArchEncodingSyntax {
    StringRef archId;
    StringRef encodingId;
    StringRef syntax;
    mlir::tblgen::Pred pred;
  };
  llvm::SmallVector<ArchEncodingSyntax> entries;
  for (const ASMStringRecord &asmStr : asmStrings) {
    mlir::tblgen::Pred pred = asmStr.getPred();
    for (const EncodedArchRecord &ea : asmStr.getArchs())
      entries.push_back({ea.getArch().getIdentifier(),
                         ea.getEncoding().getIdentifier(), asmStr.getSyntax(),
                         pred});
  }

  // Collect encoding order.
  llvm::SmallVector<StringRef> encOrder;
  llvm::StringSet<> seenEncs;
  for (const ArchEncodingSyntax &e : entries) {
    if (seenEncs.insert(e.encodingId).second)
      encOrder.push_back(e.encodingId);
  }

  // Generate the printer function.
  ctx.withSelf("_inst");
  os << "static ::mlir::LogicalResult print" << op.getCppClassName() << "(\n";
  os << "    ::mlir::aster::amdgcn::AsmPrinter &printer,\n";
  os << "    ::mlir::aster::TargetAttrInterface tgt,\n";
  os << "    ::mlir::Operation *op) {\n";
  os.indent();
  os << "auto _inst = ::llvm::cast<" << qualClass << ">(op);\n";
  os << "(void)_inst;\n";
  os << "auto _encOrFailure = _inst.getEncoding(tgt);\n";
  os << "if (::mlir::failed(_encOrFailure))\n";
  os << "  return op->emitError(\"failed to get encoding\");\n";
  os << "auto _enc = *_encOrFailure;\n";

  // Emit encoding dispatch.
  for (StringRef encId : encOrder) {
    // Collect all (arch, syntax, pred) tuples for this encoding.
    struct ArchSyntaxPred {
      StringRef archId;
      StringRef syntax;
      mlir::tblgen::Pred pred;
    };
    llvm::SmallVector<ArchSyntaxPred> archEntries;
    for (const ArchEncodingSyntax &e : entries) {
      if (e.encodingId != encId)
        continue;
      archEntries.push_back({e.archId, e.syntax, e.pred});
    }

    os << "if (_enc == ::mlir::aster::amdgcn::Encoding::" << encId << ") {\n";
    os.indent();

    // Check if all entries share the same arch, syntax, and predicate.
    bool allSame = llvm::all_of(archEntries, [&](const ArchSyntaxPred &asp) {
      return asp.syntax == archEntries.front().syntax &&
             asp.pred.getCondition() == archEntries.front().pred.getCondition();
    });

    if (allSame) {
      std::string predStr = mlir::tblgen::tgfmt(
          archEntries.front().pred.getCondition(), &ctx, "_inst");
      bool isTruePred = (predStr == "true");
      if (!isTruePred) {
        os << "if ((" << predStr << ")) {\n";
        os.indent();
      }
      emitSyntax(archEntries.front().syntax, os);
      if (!isTruePred) {
        os.unindent();
        os << "}\n";
      }
    } else {
      // Group entries by arch, preserving order.
      llvm::SmallVector<StringRef> archOrder;
      llvm::StringSet<> seenArchs;
      for (const ArchSyntaxPred &asp : archEntries) {
        if (seenArchs.insert(asp.archId).second)
          archOrder.push_back(asp.archId);
      }

      for (StringRef archId : archOrder) {
        // Collect all entries for this arch within this encoding.
        llvm::SmallVector<const ArchSyntaxPred *> archGroup;
        for (const ArchSyntaxPred &asp : archEntries) {
          if (asp.archId == archId)
            archGroup.push_back(&asp);
        }

        os << "if (tgt.getTargetFamily() == "
              "::mlir::aster::amdgcn::ISAVersion::"
           << archId << ") {\n";
        os.indent();
        for (const ArchSyntaxPred *asp : archGroup) {
          std::string predStr =
              mlir::tblgen::tgfmt(asp->pred.getCondition(), &ctx, "_inst");
          bool isTruePred = (predStr == "true");
          if (!isTruePred) {
            os << "if ((" << predStr << ")) {\n";
            os.indent();
          }
          emitSyntax(asp->syntax, os);
          if (!isTruePred) {
            os.unindent();
            os << "}\n";
          }
        }
        os.unindent();
        os << "}\n";
      }
    }
    os.unindent();
    os << "}\n";
  }

  os << "return ::mlir::failure();\n";
  os.unindent();
  os << "}\n";
  osOut << mlir::tblgen::tgfmt(strStream.str, &ctx);
}

//===----------------------------------------------------------------------===//
// Top-level generator
//===----------------------------------------------------------------------===//

static bool generateInstAsmPrinters(const llvm::RecordKeeper &records,
                                    raw_ostream &os) {
  llvm::SmallVector<const llvm::Record *> instRecs =
      llvm::to_vector(records.getAllDerivedDefinitions("AMDISAInstruction"));
  llvm::sort(instRecs, llvm::LessRecord());

  llvm::interleave(
      instRecs, os,
      [&](const llvm::Record *instRec) {
        ASMPrinterHandler handler(instRec);
        handler.genPrinter(os);
      },
      "\n");

  // Generate the opcode to printer function table.
  os << "\nstatic const llvm::SmallVector<"
        "llvm::function_ref<mlir::LogicalResult(mlir::aster::amdgcn::OpCode, "
        "mlir::aster::amdgcn::AsmPrinter &, mlir::Operation *)>> "
        "_instructionPrinters = {\n";
  os << "  nullptr, // OpCode::Invalid\n";

  // Generate each table entry.
  llvm::interleave(
      instRecs, os,
      [&](const llvm::Record *instRec) {
        AMDInst inst(instRec);
        os << "  print" << inst.getName() << ",";
      },
      "\n");
  os << "\n};\n";
  return false;
}

//===----------------------------------------------------------------------===//
// TableGen Registration
//===----------------------------------------------------------------------===//

static GenRegistration
    generateInstAsmPrintersReg("gen-inst-asm-printers",
                               "Generate encoding-aware inst asm printers",
                               generateInstAsmPrinters);
