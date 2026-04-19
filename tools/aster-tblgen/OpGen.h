//===- OpGen.h - Aster op definitions generator -----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry points for the Aster op definitions generator.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TOOLS_ASTERTBLGEN_OPGEN_H
#define ASTER_TOOLS_ASTERTBLGEN_OPGEN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace aster {

/// Emit Aster op declarations for all op records in \p defs.
bool emitOpDecls(const llvm::RecordKeeper &records,
                 llvm::ArrayRef<const llvm::Record *> defs, unsigned shardCount,
                 llvm::raw_ostream &os, bool fatalOnError = true);

/// Emit Aster op definitions for all op records in \p defs.
bool emitOpDefs(const llvm::RecordKeeper &records,
                llvm::ArrayRef<const llvm::Record *> defs, unsigned shardCount,
                llvm::raw_ostream &os, bool fatalOnError = true);

} // namespace aster

#endif // ASTER_TOOLS_ASTERTBLGEN_OPGEN_H
