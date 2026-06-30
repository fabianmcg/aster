//===- WrapInModule.cpp ---------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_WRAPINMODULE
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct WrapInModule : public amdgcn::impl::WrapInModuleBase<WrapInModule> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void WrapInModule::runOnOperation() {
  mlir::ModuleOp builtinModule = getOperation();

  std::optional<Target> targetVal = symbolizeTarget(target);
  if (!targetVal) {
    emitError(builtinModule.getLoc())
        << "unknown AMDGCN target '" << target << "'";
    return signalPassFailure();
  }

  MLIRContext *ctx = builtinModule.getContext();
  OpBuilder builder(ctx);

  // Collect all pre-existing ops before inserting the new amdgcn.module.
  SmallVector<Operation *> toMove;
  for (Operation &op : *builtinModule.getBody())
    toMove.push_back(&op);

  // Insert the new amdgcn.module at the beginning of the builtin.module.
  builder.setInsertionPointToStart(builtinModule.getBody());
  auto amdgcnModule = builder.create<amdgcn::ModuleOp>(
      builtinModule.getLoc(), TargetAttr::get(ctx, *targetVal),
      StringAttr::get(ctx, moduleName),
      /*sym_visibility=*/StringAttr{});

  // The build() helper adds an empty region; add the required entry block.
  Block *amdgcnBody = new Block();
  amdgcnModule.getBodyRegion().push_back(amdgcnBody);

  // Splice each pre-existing op into the amdgcn.module body.
  for (Operation *op : toMove)
    op->moveBefore(amdgcnBody, amdgcnBody->end());
}
