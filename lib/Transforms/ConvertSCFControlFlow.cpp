//===- ConvertSCFControlFlow.cpp - SCF to CF control flow conversion ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts SCF control flow to CF dialect with explicit basic block structure.
// Uniform conditions lower to plain cf.cond_br. Divergent scf.if, scf.for,
// and scf.while conditions are bracketed with aster_utils.get_cf_mask /
// aster_utils.set_cf_mask so that downstream codegen can insert saveexec/or
// sequences. Divergent scf.if with SSA results are rejected. The fallthrough
// layout required by amdgcn-legalize-cf is maintained by construction from
// the scf.for / scf.if / scf.while lowering patterns.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::aster {
#define GEN_PASS_DEF_CONVERTSCFCONTROLFLOW
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

/// Return i32 for wave32 targets and i64 for wave64 targets.
static Type getCfMaskType(Operation *op) {
  return amdgcn::isWave32(op) ? IntegerType::get(op->getContext(), 32)
                              : IntegerType::get(op->getContext(), 64);
}

namespace {
//===----------------------------------------------------------------------===//
// ConvertSCFControlFlow pass
//===----------------------------------------------------------------------===//

struct ConvertSCFControlFlow
    : public aster::impl::ConvertSCFControlFlowBase<ConvertSCFControlFlow> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Convert a scf.for operation to CF dialect control flow.
  LogicalResult convertForOp(scf::ForOp forOp, DataFlowSolver &solver);

  /// Lower a uniform scf.for to a plain cf.cond_br loop.
  LogicalResult convertUniformForOp(scf::ForOp forOp);

  /// Lower a divergent scf.for with exec-mask save/restore.
  LogicalResult convertDivergentForOp(scf::ForOp forOp);

  /// Convert a scf.if operation to CF dialect control flow.
  LogicalResult convertIfOp(scf::IfOp ifOp, DataFlowSolver &solver);

  /// Lower a uniform scf.if to plain cf.cond_br / cf.br.
  LogicalResult convertUniformIfOp(scf::IfOp ifOp);

  /// Lower a divergent scf.if with exec-mask save/restore.
  LogicalResult convertDivergentIfOp(scf::IfOp ifOp);

  /// Convert a scf.while operation to CF dialect control flow.
  LogicalResult convertWhileOp(scf::WhileOp whileOp, DataFlowSolver &solver);

  /// Lower a uniform scf.while to a plain cf.cond_br loop.
  LogicalResult convertUniformWhileOp(scf::WhileOp whileOp);

  /// Lower a divergent scf.while with exec-mask save/restore.
  LogicalResult convertDivergentWhileOp(scf::WhileOp whileOp);
};

/// Build the CF block structure for a scf.for op. When divergent is true,
/// get_cf_mask / set_cf_mask ops are emitted to bracket the loop. The
/// active-lane condition is threaded as a block argument into bbBody so that
/// the mask is narrowed only at the top of each iteration body and restored
/// once after the loop — no mask ops are emitted before the initial branch.
static LogicalResult buildForCfg(scf::ForOp forOp, bool divergent) {
  Location loc = forOp.getLoc();
  IRRewriter rewriter(forOp);

  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  Type ivType = forOp.getInductionVar().getType();
  Type i1Ty = IntegerType::get(forOp->getContext(), 1);

  // Capture yield operands before modifying the body.
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> yieldOperands(yieldOp.getOperands());

  Block *bbPre = forOp->getBlock();
  Block *bbEnd = rewriter.splitBlock(bbPre, std::next(forOp->getIterator()));
  Block *bbBody = rewriter.createBlock(bbEnd);

  // bbBody arguments: [iv, (activeCond : i1)?, iter_args...]
  // The activeCond bbarg carries the per-iteration active-lane condition into
  // the body so set_cf_mask can be placed at the top without a separate
  // set_cf_mask before the initial branch.
  bbBody->addArgument(ivType, loc);
  unsigned condArgIdx = 0;
  if (divergent) {
    condArgIdx = bbBody->getNumArguments();
    bbBody->addArgument(i1Ty, loc);
  }
  for (Value iterArg : forOp.getRegionIterArgs())
    bbBody->addArgument(iterArg.getType(), loc);

  for (Value result : forOp.getResults())
    bbEnd->addArgument(result.getType(), loc);

  rewriter.setInsertionPoint(forOp);
  Value saved;
  if (divergent)
    saved =
        aster_utils::GetCfMaskOp::create(rewriter, loc, getCfMaskType(forOp))
            .getMask();

  // Initial comparison; the result is passed as condArg on the true edge.
  Value initCond = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, lowerBound, upperBound);

  // Build initial branch operands: [lowerBound, (initCond)?, init_args...]
  SmallVector<Value> initBranchArgs = {lowerBound};
  if (divergent)
    initBranchArgs.push_back(initCond);
  initBranchArgs.append(forOp.getInitArgs().begin(), forOp.getInitArgs().end());

  cf::CondBranchOp::create(rewriter, loc, initCond, bbBody, initBranchArgs,
                           bbEnd, forOp.getInitArgs());

  rewriter.eraseOp(forOp.getBody()->getTerminator());
  Value ivBlockArg = bbBody->getArgument(0);

  // Offset of iter_args in bbBody: after iv and optional condArg.
  unsigned iterArgBase = divergent ? 2 : 1;
  int64_t numIterArgs = static_cast<int64_t>(forOp.getNumRegionIterArgs());

  IRMapping blockArgMapping;
  blockArgMapping.map(forOp.getInductionVar(), ivBlockArg);
  for (int64_t i = 0; i < numIterArgs; ++i)
    blockArgMapping.map(
        forOp.getRegionIterArgs()[i],
        bbBody->getArgument(iterArgBase + static_cast<unsigned>(i)));

  // Narrow EXEC to the active lanes for this iteration at the top of the body.
  if (divergent) {
    rewriter.setInsertionPointToStart(bbBody);
    Value condArg = bbBody->getArgument(condArgIdx);
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, condArg, UnitAttr{});
  }

  rewriter.setInsertionPointToEnd(bbBody);
  SmallVector<Value> bodyArgMapping = {ivBlockArg};
  bodyArgMapping.append(bbBody->args_begin() + iterArgBase,
                        bbBody->args_begin() + iterArgBase +
                            static_cast<unsigned>(numIterArgs));
  rewriter.inlineBlockBefore(forOp.getBody(), bbBody, bbBody->end(),
                             bodyArgMapping);

  for (Value &val : yieldOperands)
    val = blockArgMapping.lookupOrDefault(val);

  Value ivNext = arith::AddIOp::create(rewriter, loc, ivBlockArg, step);
  Value backEdgeCond = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, ivNext, upperBound);

  // Back-edge operands: [ivNext, (backEdgeCond)?, yield_vals...]
  SmallVector<Value> backEdgeArgs = {ivNext};
  if (divergent)
    backEdgeArgs.push_back(backEdgeCond);
  backEdgeArgs.append(yieldOperands);

  cf::CondBranchOp::create(rewriter, loc, backEdgeCond, bbBody, backEdgeArgs,
                           bbEnd, yieldOperands);

  // Restore the saved mask once after the loop.
  if (divergent) {
    rewriter.setInsertionPointToStart(bbEnd);
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, Value{}, UnitAttr{});
  }

  rewriter.replaceOp(forOp, bbEnd->getArguments());
  return success();
}

LogicalResult ConvertSCFControlFlow::convertForOp(scf::ForOp forOp,
                                                  DataFlowSolver &solver) {
  // A for-loop is divergent if any of its bounds (lb, ub, step) is divergent.
  // Treat a missing lattice or non-uniform state pessimistically as divergent.
  auto isUniform = [&](Value v) {
    const aster::dataflow::ThreadUniformLattice *lattice =
        solver.lookupState<aster::dataflow::ThreadUniformLattice>(v);
    return lattice && lattice->getValue().isUniform();
  };
  bool uniform = isUniform(forOp.getLowerBound()) &&
                 isUniform(forOp.getUpperBound()) && isUniform(forOp.getStep());
  if (uniform)
    return convertUniformForOp(forOp);
  return convertDivergentForOp(forOp);
}

LogicalResult ConvertSCFControlFlow::convertUniformForOp(scf::ForOp forOp) {
  return buildForCfg(forOp, /*divergent=*/false);
}

LogicalResult ConvertSCFControlFlow::convertDivergentForOp(scf::ForOp forOp) {
  return buildForCfg(forOp, /*divergent=*/true);
}

LogicalResult ConvertSCFControlFlow::convertIfOp(scf::IfOp ifOp,
                                                 DataFlowSolver &solver) {
  Value condition = ifOp.getCondition();
  // Treat missing lattice or non-uniform state pessimistically as divergent.
  const aster::dataflow::ThreadUniformLattice *lattice =
      solver.lookupState<aster::dataflow::ThreadUniformLattice>(condition);
  bool isUniform = lattice && lattice->getValue().isUniform();

  if (!isUniform) {
    // Reject divergent if with results before any IR mutation.
    if (ifOp.getNumResults() != 0)
      return ifOp.emitError()
             << "divergent scf.if with results is not supported";
    return convertDivergentIfOp(ifOp);
  }
  return convertUniformIfOp(ifOp);
}

LogicalResult ConvertSCFControlFlow::convertUniformIfOp(scf::IfOp ifOp) {
  Location loc = ifOp.getLoc();
  IRRewriter rewriter(ifOp);
  Value condition = ifOp.getCondition();
  bool hasElse = !ifOp.getElseRegion().empty();

  // Capture yield operands before erasing the terminators.
  Block *thenBlock = ifOp.thenBlock();
  auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
  SmallVector<Value> thenYieldOperands(thenYield.getOperands());
  rewriter.eraseOp(thenYield);

  Block *elseBlock = nullptr;
  SmallVector<Value> elseYieldOperands;
  if (hasElse) {
    elseBlock = ifOp.elseBlock();
    auto elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());
    elseYieldOperands.assign(elseYield.getOperands().begin(),
                             elseYield.getOperands().end());
    rewriter.eraseOp(elseYield);
  }

  Block *bbMerge =
      rewriter.splitBlock(ifOp->getBlock(), std::next(ifOp->getIterator()));
  Block *bbThen = rewriter.createBlock(bbMerge);
  Block *bbElse = hasElse ? rewriter.createBlock(bbMerge) : bbMerge;

  for (Value result : ifOp.getResults())
    bbMerge->addArgument(result.getType(), loc);

  rewriter.setInsertionPoint(ifOp);
  cf::CondBranchOp::create(rewriter, loc, condition, bbThen, ValueRange(),
                           bbElse, ValueRange());

  rewriter.inlineBlockBefore(thenBlock, bbThen, bbThen->end());
  rewriter.setInsertionPointToEnd(bbThen);
  cf::BranchOp::create(rewriter, loc, bbMerge, thenYieldOperands);

  if (hasElse) {
    rewriter.inlineBlockBefore(elseBlock, bbElse, bbElse->end());
    rewriter.setInsertionPointToEnd(bbElse);
    cf::BranchOp::create(rewriter, loc, bbMerge, elseYieldOperands);
  }

  rewriter.replaceOp(ifOp, bbMerge->getArguments());
  return success();
}

LogicalResult ConvertSCFControlFlow::convertDivergentIfOp(scf::IfOp ifOp) {
  Location loc = ifOp.getLoc();
  IRRewriter rewriter(ifOp);
  Value condition = ifOp.getCondition();
  bool hasElse = !ifOp.getElseRegion().empty();

  // Capture block pointers and erase yields before moving blocks.
  Block *thenBlock = ifOp.thenBlock();
  rewriter.eraseOp(thenBlock->getTerminator());

  Block *elseBlock = nullptr;
  if (hasElse) {
    elseBlock = ifOp.elseBlock();
    rewriter.eraseOp(elseBlock->getTerminator());
  }

  Block *bbMerge =
      rewriter.splitBlock(ifOp->getBlock(), std::next(ifOp->getIterator()));
  Block *bbThen = rewriter.createBlock(bbMerge);
  Block *bbElse = hasElse ? rewriter.createBlock(bbMerge) : bbMerge;

  Type maskTy = getCfMaskType(ifOp);

  // Save the current EXEC mask before branching. Narrow to then-condition
  // lanes before the branch so that the canonicalizer cannot fold the
  // condition to a constant inside bbThen (it would do so because bbThen is
  // only reachable when condition = true).
  rewriter.setInsertionPoint(ifOp);
  Value saved =
      aster_utils::GetCfMaskOp::create(rewriter, loc, maskTy).getMask();
  aster_utils::SetCfMaskOp::create(rewriter, loc, saved, condition,
                                   /*complement=*/UnitAttr{});
  cf::CondBranchOp::create(rewriter, loc, condition, bbThen, ValueRange(),
                           bbElse, ValueRange());

  rewriter.inlineBlockBefore(thenBlock, bbThen, bbThen->end());
  rewriter.setInsertionPointToEnd(bbThen);
  aster_utils::SetCfMaskOp::create(rewriter, loc, saved, Value{}, UnitAttr{});
  cf::BranchOp::create(rewriter, loc, hasElse ? bbElse : bbMerge, ValueRange());

  if (hasElse) {
    // Narrow EXEC to else-condition lanes (complement) at the start of else.
    // bbElse has two predecessors (bbPre false edge and bbThen back-edge) so
    // the canonicalizer will not constant-fold condition here.
    rewriter.setInsertionPointToStart(bbElse);
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, condition,
                                     rewriter.getUnitAttr());

    rewriter.inlineBlockBefore(elseBlock, bbElse, bbElse->end());
    rewriter.setInsertionPointToEnd(bbElse);
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, Value{}, UnitAttr{});
    cf::BranchOp::create(rewriter, loc, bbMerge, ValueRange());
  } else {
    // Without an else block, the false edge jumps directly to bbMerge. Restore
    // the saved EXEC mask at the top of bbMerge so the skip path does not leave
    // EXEC narrowed for subsequent code.
    rewriter.setInsertionPointToStart(bbMerge);
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, Value{}, UnitAttr{});
  }

  rewriter.eraseOp(ifOp);
  return success();
}

/// Inline the scf.while before region into bbBefore, restoring the saved mask
/// first when divergent, and emit the condition branch.
/// When divergent, the mask is narrowed to active-condition lanes before the
/// branch so that the after block executes only the true-condition lanes.
static void buildWhileBeforeBlock(IRRewriter &rewriter, Location loc,
                                  Block *bbBefore, Block *bbAfter, Block *bbEnd,
                                  Block *beforeBlock, Value cond,
                                  ArrayRef<Value> fwdOperands, Value saved,
                                  bool divergent) {
  // Build a mapping from the original before-region args to bbBefore args.
  IRMapping beforeMap;
  for (int64_t i = 0, n = static_cast<int64_t>(beforeBlock->getNumArguments());
       i < n; ++i)
    beforeMap.map(beforeBlock->getArgument(static_cast<unsigned>(i)),
                  bbBefore->getArgument(static_cast<unsigned>(i)));

  // Restore the mask as the first op so the condition sees all live threads.
  if (divergent) {
    rewriter.setInsertionPointToStart(bbBefore);
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, Value{}, UnitAttr{});
  }

  // Inline the before region.
  rewriter.setInsertionPointToEnd(bbBefore);
  rewriter.inlineBlockBefore(beforeBlock, bbBefore, bbBefore->end(),
                             bbBefore->getArguments());

  // Remap the captured condition and forwarded operands through beforeMap so
  // that any values that were block arguments are resolved to bbBefore args.
  cond = beforeMap.lookupOrDefault(cond);
  SmallVector<Value> remappedFwd(fwdOperands);
  for (Value &val : remappedFwd)
    val = beforeMap.lookupOrDefault(val);

  // Narrow EXEC to active-condition lanes before the branch. This ensures only
  // true-condition lanes execute the loop body. The cf.cond_br then acts as a
  // scalar EXECZ check: if all lanes exited, jump to bbEnd.
  rewriter.setInsertionPointToEnd(bbBefore);
  if (divergent)
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, cond, UnitAttr{});

  // Terminate bbBefore: true -> body, false -> exit.
  rewriter.setInsertionPointToEnd(bbBefore);
  cf::CondBranchOp::create(rewriter, loc, cond, bbAfter, remappedFwd, bbEnd,
                           remappedFwd);
}

/// Inline the scf.while after region into bbAfter and emit the back-edge.
static void buildWhileAfterBlock(IRRewriter &rewriter, Location loc,
                                 Block *bbBefore, Block *bbAfter,
                                 Block *afterBlock,
                                 ArrayRef<Value> yieldOperands) {
  // Build a mapping from the original after-region args to bbAfter args.
  IRMapping afterMap;
  for (int64_t i = 0, n = static_cast<int64_t>(afterBlock->getNumArguments());
       i < n; ++i)
    afterMap.map(afterBlock->getArgument(static_cast<unsigned>(i)),
                 bbAfter->getArgument(static_cast<unsigned>(i)));

  // Inline the after region.
  rewriter.setInsertionPointToEnd(bbAfter);
  rewriter.inlineBlockBefore(afterBlock, bbAfter, bbAfter->end(),
                             bbAfter->getArguments());

  // Remap yield operands through afterMap and emit the back-edge to bbBefore.
  SmallVector<Value> remappedYield(yieldOperands);
  for (Value &val : remappedYield)
    val = afterMap.lookupOrDefault(val);

  rewriter.setInsertionPointToEnd(bbAfter);
  cf::BranchOp::create(rewriter, loc, bbBefore, remappedYield);
}

/// Create the four CFG blocks for a lowered scf.while in physical order
/// (bbPre, bbBefore, bbAfter, bbEnd) and populate their block arguments.
static void buildWhileBlocks(IRRewriter &rewriter, scf::WhileOp whileOp,
                             Block *&bbBefore, Block *&bbAfter, Block *&bbEnd) {
  Location loc = whileOp.getLoc();
  Block *bbPre = whileOp->getBlock();
  bbEnd = rewriter.splitBlock(bbPre, std::next(whileOp->getIterator()));
  bbBefore = rewriter.createBlock(bbEnd);
  bbAfter = rewriter.createBlock(bbEnd);

  for (BlockArgument arg : whileOp.getBeforeArguments())
    bbBefore->addArgument(arg.getType(), loc);
  for (BlockArgument arg : whileOp.getAfterArguments())
    bbAfter->addArgument(arg.getType(), loc);
  for (Value result : whileOp.getResults())
    bbEnd->addArgument(result.getType(), loc);
}

/// Build the CF block structure for a scf.while op. When divergent is true,
/// get_cf_mask / set_cf_mask ops are emitted to bracket the loop.
static LogicalResult buildWhileCfg(scf::WhileOp whileOp, bool divergent) {
  Location loc = whileOp.getLoc();
  IRRewriter rewriter(whileOp);

  // Capture terminators and operands before any IR mutation.
  scf::ConditionOp condOp = whileOp.getConditionOp();
  Value cond = condOp.getCondition();
  SmallVector<Value> fwdOperands(condOp.getArgs());

  scf::YieldOp yieldOp = whileOp.getYieldOp();
  SmallVector<Value> yieldOperands(yieldOp.getOperands());

  Block *beforeBlock = whileOp.getBeforeBody();
  Block *afterBlock = whileOp.getAfterBody();

  Block *bbBefore = nullptr;
  Block *bbAfter = nullptr;
  Block *bbEnd = nullptr;
  buildWhileBlocks(rewriter, whileOp, bbBefore, bbAfter, bbEnd);

  // Erase the region terminators before inlining.
  rewriter.eraseOp(condOp);
  rewriter.eraseOp(yieldOp);

  // Build the entry branch in bbPre. Save the mask first when divergent.
  rewriter.setInsertionPoint(whileOp);
  Value saved;
  if (divergent)
    saved =
        aster_utils::GetCfMaskOp::create(rewriter, loc, getCfMaskType(whileOp))
            .getMask();
  cf::BranchOp::create(rewriter, loc, bbBefore, whileOp.getInits());

  buildWhileBeforeBlock(rewriter, loc, bbBefore, bbAfter, bbEnd, beforeBlock,
                        cond, fwdOperands, saved, divergent);
  buildWhileAfterBlock(rewriter, loc, bbBefore, bbAfter, afterBlock,
                       yieldOperands);

  // Restore the pre-loop mask as the first op of bbEnd when divergent.
  if (divergent) {
    rewriter.setInsertionPointToStart(bbEnd);
    aster_utils::SetCfMaskOp::create(rewriter, loc, saved, Value{}, UnitAttr{});
  }

  rewriter.replaceOp(whileOp, bbEnd->getArguments());
  return success();
}

LogicalResult ConvertSCFControlFlow::convertWhileOp(scf::WhileOp whileOp,
                                                    DataFlowSolver &solver) {
  Value condition = whileOp.getConditionOp().getCondition();
  // Treat missing lattice or non-uniform state pessimistically as divergent.
  const aster::dataflow::ThreadUniformLattice *lattice =
      solver.lookupState<aster::dataflow::ThreadUniformLattice>(condition);
  bool isUniform = lattice && lattice->getValue().isUniform();
  if (isUniform)
    return convertUniformWhileOp(whileOp);
  return convertDivergentWhileOp(whileOp);
}

LogicalResult
ConvertSCFControlFlow::convertUniformWhileOp(scf::WhileOp whileOp) {
  return buildWhileCfg(whileOp, /*divergent=*/false);
}

LogicalResult
ConvertSCFControlFlow::convertDivergentWhileOp(scf::WhileOp whileOp) {
  return buildWhileCfg(whileOp, /*divergent=*/true);
}

void ConvertSCFControlFlow::runOnOperation() {
  Operation *op = getOperation();

  // Configure and run the dataflow solver once over the original IR.
  DataFlowSolver solver;
  mlir::dataflow::loadBaselineAnalyses(solver);
  solver.load<aster::dataflow::ThreadUniformAnalysis>();
  if (failed(solver.initializeAndRun(op))) {
    signalPassFailure();
    return;
  }

  // Collect all SCF operations first to avoid modifying while iterating.
  // Walk is post-order (inner before outer), but we need top-down order
  // (outer before inner) so that converting an outer op inlines the body
  // while inner SCF ops remain intact for later conversion.
  SmallVector<Operation *> scfOps;
  op->walk([&](Operation *nestedOp) {
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(nestedOp))
      scfOps.push_back(nestedOp);
  });
  std::reverse(scfOps.begin(), scfOps.end());

  // Convert each SCF operation. The condition Value objects remain valid in
  // the solver state even after the outer op is inlined, so a single solver
  // run is sufficient for all ops.
  for (Operation *scfOp : scfOps) {
    LogicalResult result = success();
    if (auto forOp = dyn_cast<scf::ForOp>(scfOp))
      result = convertForOp(forOp, solver);
    else if (auto ifOp = dyn_cast<scf::IfOp>(scfOp))
      result = convertIfOp(ifOp, solver);
    else if (auto whileOp = dyn_cast<scf::WhileOp>(scfOp))
      result = convertWhileOp(whileOp, solver);
    if (failed(result)) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace
