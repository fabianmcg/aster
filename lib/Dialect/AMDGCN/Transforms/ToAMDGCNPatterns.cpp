//===- ToAMDGCNPatterns.cpp -----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert to AMDGCN patterns to more complex ops that are too complex for PDLL.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/ValueOrConst.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "llvm/ADT/bit.h"

#include <cstdint>
#include <utility>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

/// Create an SCC alloca value for use as scc_dst or scc_src operand.
static Value createSCCAlloca(OpBuilder &builder, Location loc) {
  return AllocaOp::create(builder, loc,
                          SCCType::get(builder.getContext(), Register(0)));
}

/// Helper to create an SOP2Out2In2 instruction and return the dst result.
/// These instructions have 2 outs (dst0 + scc_dst) and 2 ins (src0, src1).
template <typename OpTy>
static Value createSOP2Out2In2(OpBuilder &builder, Location loc, Value dst,
                               Value src0, Value src1) {
  Value sccDst = createSCCAlloca(builder, loc);
  return OpTy::create(builder, loc, dst, sccDst, src0, src1).getDst0Res();
}

/// Helper to create an SOP2Out2In3 instruction and return the dst result.
/// These instructions have 2 outs (dst0 + scc_dst) and 3 ins (src0, src1,
/// scc_src). The scc_src alloca is created internally to represent the SCC
/// carry input.
template <typename OpTy>
static Value createSOP2Out2In3(OpBuilder &builder, Location loc, Value dst,
                               Value src0, Value src1) {
  Value sccDst = createSCCAlloca(builder, loc);
  Value sccSrc = createSCCAlloca(builder, loc);
  return OpTy::create(builder, loc, dst, sccDst, src0, src1, sccSrc)
      .getDst0Res();
}

/// Lane-mask AND/OR/XOR -> s_{and,or,xor}_b32/b64 by mask width.
template <typename SOp32, typename SOp64>
static LogicalResult lowerVCCBitwise(Operation *op, PatternRewriter &rewriter,
                                     Value dst, Value lhs, Value rhs) {
  Value result;
  if (isa<VCCLoType>(dst.getType()))
    result = createSOP2Out2In2<SOp32>(rewriter, op->getLoc(), dst, lhs, rhs);
  else if (isa<VCCType>(dst.getType()))
    result = createSOP2Out2In2<SOp64>(rewriter, op->getLoc(), dst, lhs, rhs);
  else
    return failure();
  rewriter.replaceOp(op, result);
  return success();
}

namespace {
//===----------------------------------------------------------------------===//
// AddIOpPattern
//===----------------------------------------------------------------------===//

struct AddFOpPattern : public OpRewritePattern<lsir::AddFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AddFOp op,
                                PatternRewriter &rewriter) const override;
};

struct AddIOpPattern : public OpRewritePattern<lsir::AddIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AddIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// AllocaOpPattern
//===----------------------------------------------------------------------===//

struct AllocaOpPattern : public OpRewritePattern<lsir::AllocaOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AllocaOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// AssumeNoaliasOpPattern
//===----------------------------------------------------------------------===//

struct AssumeNoaliasOpPattern : public OpRewritePattern<lsir::AssumeNoaliasOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AssumeNoaliasOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// AndIOpPattern
//===----------------------------------------------------------------------===//

struct AndIOpPattern : public OpRewritePattern<lsir::AndIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AndIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// KernelOpPattern
//===----------------------------------------------------------------------===//

struct KernelOpPattern : public OpInterfaceRewritePattern<FunctionOpInterface> {
  using Base::Base;
  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// LoadOpPattern
//===----------------------------------------------------------------------===//

struct LoadOpPattern : public OpRewritePattern<lsir::LoadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::LoadOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// MulIOpPattern
//===----------------------------------------------------------------------===//

struct MaximumFOpPattern : public OpRewritePattern<lsir::MaximumFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MaximumFOp op,
                                PatternRewriter &rewriter) const override;
};

struct MinimumFOpPattern : public OpRewritePattern<lsir::MinimumFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MinimumFOp op,
                                PatternRewriter &rewriter) const override;
};

struct MulFOpPattern : public OpRewritePattern<lsir::MulFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MulFOp op,
                                PatternRewriter &rewriter) const override;
};

struct MulIOpPattern : public OpRewritePattern<lsir::MulIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MulIOp op,
                                PatternRewriter &rewriter) const override;
};

struct MulHiSIOpPattern : public OpRewritePattern<lsir::MulHiSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MulHiSIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// MovOpPattern
//===----------------------------------------------------------------------===//

struct MovOpPattern : public OpRewritePattern<lsir::MovOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MovOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// OrIOpPattern
//===----------------------------------------------------------------------===//

struct OrIOpPattern : public OpRewritePattern<lsir::OrIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::OrIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// XOrIOpPattern
//===----------------------------------------------------------------------===//

struct XOrIOpPattern : public OpRewritePattern<lsir::XOrIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::XOrIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// RegCastOpPattern
//===----------------------------------------------------------------------===//

struct RegCastOpPattern : public OpRewritePattern<lsir::RegCastOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::RegCastOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ReturnOpPattern
//===----------------------------------------------------------------------===//

struct ReturnOpPattern : public OpRewritePattern<func::ReturnOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ShLIOpPattern
//===----------------------------------------------------------------------===//

struct ShLIOpPattern : public OpRewritePattern<lsir::ShLIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ShLIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ExtSIOpPattern
//===----------------------------------------------------------------------===//

struct ExtSIOpPattern : public OpRewritePattern<lsir::ExtSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ExtSIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ExtUIOpPattern
//===----------------------------------------------------------------------===//

struct ExtUIOpPattern : public OpRewritePattern<lsir::ExtUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ExtUIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ExtFOpPattern
//===----------------------------------------------------------------------===//

struct ExtFOpPattern : public OpRewritePattern<lsir::ExtFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ExtFOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// TruncFOpPattern
//===----------------------------------------------------------------------===//

struct TruncFOpPattern : public OpRewritePattern<lsir::TruncFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::TruncFOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SIToFPOpPattern
//===----------------------------------------------------------------------===//

struct SIToFPOpPattern : public OpRewritePattern<lsir::SIToFPOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::SIToFPOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// UIToFPOpPattern
//===----------------------------------------------------------------------===//

struct UIToFPOpPattern : public OpRewritePattern<lsir::UIToFPOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::UIToFPOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// FPToSIOpPattern
//===----------------------------------------------------------------------===//

struct FPToSIOpPattern : public OpRewritePattern<lsir::FPToSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::FPToSIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// FPToUIOpPattern
//===----------------------------------------------------------------------===//

struct FPToUIOpPattern : public OpRewritePattern<lsir::FPToUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::FPToUIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// TruncIOpPattern
//===----------------------------------------------------------------------===//

struct TruncIOpPattern : public OpRewritePattern<lsir::TruncIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::TruncIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ShRSIOpPattern
//===----------------------------------------------------------------------===//

struct ShRSIOpPattern : public OpRewritePattern<lsir::ShRSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ShRSIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ShRUIOpPattern
//===----------------------------------------------------------------------===//

struct ShRUIOpPattern : public OpRewritePattern<lsir::ShRUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ShRUIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// StoreOpPattern
//===----------------------------------------------------------------------===//

struct StoreOpPattern : public OpRewritePattern<lsir::StoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SubIOpPattern
//===----------------------------------------------------------------------===//

struct SubFOpPattern : public OpRewritePattern<lsir::SubFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::SubFOp op,
                                PatternRewriter &rewriter) const override;
};

struct SubIOpPattern : public OpRewritePattern<lsir::SubIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::SubIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// TimingStartOpPattern
//===----------------------------------------------------------------------===//

struct TimingStartOpPattern : public OpRewritePattern<lsir::TimingStartOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::TimingStartOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// TimingStopOpPattern
//===----------------------------------------------------------------------===//

struct TimingStopOpPattern : public OpRewritePattern<lsir::TimingStopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::TimingStopOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

struct CmpIOpPattern : public OpRewritePattern<lsir::CmpIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::CmpIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SelectOpPattern
//===----------------------------------------------------------------------===//

struct SelectOpPattern : public OpRewritePattern<lsir::SelectOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::SelectOp op,
                                PatternRewriter &rewriter) const override;
};

struct PtrAddOpPattern : public OpRewritePattern<PtrAddOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(PtrAddOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// DivUI/RemUI/DivSI/RemSI OpPatterns
//===----------------------------------------------------------------------===//

struct DivUIOpPattern : public OpRewritePattern<lsir::DivUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::DivUIOp op,
                                PatternRewriter &rewriter) const override;
};
struct RemUIOpPattern : public OpRewritePattern<lsir::RemUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::RemUIOp op,
                                PatternRewriter &rewriter) const override;
};
struct DivSIOpPattern : public OpRewritePattern<lsir::DivSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::DivSIOp op,
                                PatternRewriter &rewriter) const override;
};
struct RemSIOpPattern : public OpRewritePattern<lsir::RemSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::RemSIOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Check if the given width is contained in the list of widths.
static bool hasWidth(unsigned width, ArrayRef<unsigned> widths) {
  return llvm::is_contained(widths, width);
}

/// Check if the given operand kind is contained in the list of kinds.
static bool isOperand(OperandKind value, ArrayRef<OperandKind> kinds) {
  return llvm::is_contained(kinds, value);
}

/// Check if the given type is a valid operand of the given kind and size.
static bool isValidOperand(Type type, ArrayRef<OperandKind> kind,
                           int16_t numWords) {
  OperandKind operandKind = getOperandKind(type);
  if (auto rT = dyn_cast<RegisterTypeInterface>(type)) {
    if (!llvm::is_contained(kind, operandKind))
      return false;
    return rT.getAsRange().size() == numWords;
  }
  return llvm::is_contained(kind, operandKind);
}
static bool isValidOperand(Type type, OperandKind kind, int16_t numWords) {
  return (kind == OperandKind::SGPR &&
          isValidOperand(type, {OperandKind::SGPR, OperandKind::IntImm},
                         numWords)) ||
         (kind == OperandKind::VGPR &&
          isValidOperand(
              type, {OperandKind::SGPR, OperandKind::VGPR, OperandKind::IntImm},
              numWords));
}

/// Helper function to get element from range or default value.
static Value getElemOr(ValueRange range, int32_t i, Value value) {
  if (range.empty())
    return value;
  return range[i];
}

/// Create an i32 constant value.
static Value getI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getIntegerAttr(builder.getI32Type(), value));
}

// Create a new-style VOP instruction with 1 output and 2 inputs.
template <typename OpT>
static Value createNewVOP(OpBuilder &builder, Location loc, Value dst,
                          Value src0, Value src1) {
  return OpT::create(builder, loc, dst, src0, src1).getDst0Res();
}

/// If the destination is SGPR but any source operand is VGPR (mixed uniform/
/// dynamic from assume_uniform), override `kind` to VGPR and set
/// `sgprDestVgprSrc` so the caller can copy the VGPR result back to SGPR.
static void handleMixedSgprVgprDest(OperandKind &kind,
                                    RegisterTypeInterface &oTy, Value lhs,
                                    Value rhs, bool &sgprDestVgprSrc,
                                    MLIRContext *ctx) {
  sgprDestVgprSrc = false;
  if (kind != OperandKind::SGPR)
    return;
  OperandKind lhsK = getOperandKind(lhs.getType());
  OperandKind rhsK = getOperandKind(rhs.getType());
  if (lhsK != OperandKind::VGPR && rhsK != OperandKind::VGPR)
    return;
  sgprDestVgprSrc = true;
  kind = OperandKind::VGPR;
  // Derive VGPR range size from original SGPR type to handle 32- and 64-bit.
  int16_t rangeSize = oTy.getAsRange().size();
  oTy = cast<RegisterTypeInterface>(getVGPR(ctx, rangeSize));
}

/// After a VGPR-path computation for a mixed SGPR dest, copy the VGPR result
/// to the SGPR destination and return the SGPR value.
static Value finishMixedSgprVgprDest(PatternRewriter &rewriter, Location loc,
                                     Value dst, Value vgprResult,
                                     bool sgprDestVgprSrc) {
  if (!sgprDestVgprSrc)
    return vgprResult;
  return lsir::CopyOp::create(rewriter, loc, dst, vgprResult).getTargetRes();
}

/// Check validity of an AMDGCN arith op.
static LogicalResult checkAIOp(Operation *op, PatternRewriter &rewriter,
                               OperandKind kind, Value lhs, Value rhs,
                               RegisterTypeInterface oTy, unsigned width,
                               OperandKind &lhsKind, OperandKind &rhsKind,
                               ArrayRef<unsigned> sgprWidths,
                               ArrayRef<unsigned> vgprWidths) {
  // Check that the output type is an AMDGCN register type
  if (!isAMDReg(oTy)) {
    return rewriter.notifyMatchFailure(
        op, "operand type is not an AMDGCN register type");
  }

  // AGPRs are not supported for arith operations
  if (kind == OperandKind::AGPR)
    return rewriter.notifyMatchFailure(op, "operand type cannot be AGPR");
  int16_t rangeSize = oTy.getAsRange().size();

  if (rangeSize != ((width / 8) + 3) / 4) {
    return rewriter.notifyMatchFailure(
        op, "register range size does not match the operation width");
  }

  // Validate supported widths
  if (kind == OperandKind::SGPR && !hasWidth(width, sgprWidths)) {
    return rewriter.notifyMatchFailure(
        op, "SGPR arith operations only support 32 or 64-bit widths");
  }
  if (kind == OperandKind::VGPR && !hasWidth(width, vgprWidths)) {
    return rewriter.notifyMatchFailure(
        op, "VGPR arith operations only support 16, 32, or 64-bit widths");
  }

  // Validate lhs and rhs operand types
  lhsKind = getOperandKind(lhs.getType());
  if (!isValidOperand(lhs.getType(), kind, rangeSize)) {
    return rewriter.notifyMatchFailure(
        op, "Invalid lhs operand type for arith operation");
  }
  rhsKind = getOperandKind(rhs.getType());
  if (!isValidOperand(rhs.getType(), kind, rangeSize)) {
    return rewriter.notifyMatchFailure(
        op, "Invalid rhs operand type for arith operation");
  }

  // Both operands shouldn't be immediates
  if (lhsKind == OperandKind::IntImm && rhsKind == OperandKind::IntImm) {
    return rewriter.notifyMatchFailure(
        op, "Expected at least one non-immediate operand for add operation");
  }
  return success();
}

static MLIRContext *getCtx(PatternRewriter &rewriter) {
  return rewriter.getContext();
}

//===----------------------------------------------------------------------===//
// AddIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult AddIOpPattern::matchAndRewrite(lsir::AddIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // If dest is SGPR but a source is VGPR (mixed uniform/dynamic), use VGPR
  // path and copy result back. Override kind/oTy for validation.
  bool sgprDestVgprSrc = false;
  handleMixedSgprVgprDest(kind, oTy, lhs, rhs, sgprDestVgprSrc,
                          op.getContext());

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();
  // Maybe split operands if they are register ranges
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsR, rhsR);
    std::swap(lhsKind, rhsKind);
  }

  // At this point, operands are valid - create the appropriate add op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SAddU32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value lo = createSOP2Out2In2<SAddU32>(
        rewriter, loc, getElemOr(dstR, 0, dst), getElemOr(lhsR, 0, lhs),
        getElemOr(rhsR, 0, rhs));
    Value hi = createSOP2Out2In3<SAddcU32>(
        rewriter, loc, getElemOr(dstR, 1, dst), getElemOr(lhsR, 1, lhs),
        getElemOr(rhsR, 1, rhs));
    rewriter.replaceOp(
        op, MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
    return success();
  }

  // Allocate VGPR temp if we're writing to SGPR from VGPR path.
  Value actualDst = dst;
  if (sgprDestVgprSrc) {
    int16_t rangeSize = oTy.getAsRange().size();
    actualDst = createAllocation(rewriter, loc,
                                 getVGPR(rewriter.getContext(), rangeSize));
  }

  // Handle the VGPR case
  Value result;
  if (width <= 16) {
    result = createNewVOP<VAddU16>(rewriter, loc, actualDst, lhs, rhs);
  } else if (width <= 32) {
    result = createNewVOP<VAddU32>(rewriter, loc, actualDst, lhs, rhs);
  } else {
    // 64-bit VGPR add
    result = VLshlAddU64::create(rewriter, loc, actualDst, lhs,
                                 getI32Constant(rewriter, loc, 0), rhs)
                 .getDst0Res();
  }

  // Copy VGPR result back to SGPR destination.
  result = finishMixedSgprVgprDest(rewriter, loc, dst, result, sgprDestVgprSrc);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// Float binary op patterns (AddF, SubF, MulF, MaximumF, MinimumF)
//===----------------------------------------------------------------------===//

/// Generic lowering for lsir binary float ops -> new VOP instructions.
/// Float ops are always VGPR (no SGPR float arithmetic on AMD GPUs).
template <typename LsirOp, typename VOp32, typename VOp16>
static LogicalResult lowerBinaryFloatOp(LsirOp op, PatternRewriter &rewriter) {
  RegisterTypeInterface oTy = op.getDst().getType();
  OperandKind kind = getOperandKind(oTy);
  if (kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "float ops require VGPR dest");

  unsigned width = op.getSemantics().getWidth();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Location loc = op.getLoc();

  OperandKind rhsKind = getOperandKind(rhs.getType());

  // Commute if rhs is immediate/SGPR (VOP2 src0 can be SGPR, src1 must be
  // VGPR).
  OperandKind lhsKind = getOperandKind(lhs.getType());
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsKind, rhsKind);
  }

  if (width <= 16) {
    Value result = createNewVOP<VOp16>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width <= 32) {
    Value result = createNewVOP<VOp32>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "64-bit float VOP not supported");
}

LogicalResult AddFOpPattern::matchAndRewrite(lsir::AddFOp op,
                                             PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::AddFOp, VAddF32, VAddF16>(op, rewriter);
}

LogicalResult SubFOpPattern::matchAndRewrite(lsir::SubFOp op,
                                             PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::SubFOp, VSubF32, VSubF32>(op, rewriter);
}

LogicalResult MulFOpPattern::matchAndRewrite(lsir::MulFOp op,
                                             PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::MulFOp, VMulF32, VMulF16>(op, rewriter);
}

LogicalResult
MaximumFOpPattern::matchAndRewrite(lsir::MaximumFOp op,
                                   PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::MaximumFOp, VMaxF32, VMaxF32>(op, rewriter);
}

LogicalResult
MinimumFOpPattern::matchAndRewrite(lsir::MinimumFOp op,
                                   PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::MinimumFOp, VMinF32, VMinF32>(op, rewriter);
}

//===----------------------------------------------------------------------===//
// AllocaOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
AllocaOpPattern::matchAndRewrite(lsir::AllocaOp op,
                                 PatternRewriter &rewriter) const {
  // Check that the output type is an AMDGCN register type
  if (!isAMDReg(op.getType())) {
    return rewriter.notifyMatchFailure(
        op, "operand type is not an AMDGCN register type");
  }
  rewriter.replaceOp(op, createAllocation(rewriter, op.getLoc(), op.getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// AssumeNoaliasOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
AssumeNoaliasOpPattern::matchAndRewrite(lsir::AssumeNoaliasOp op,
                                        PatternRewriter &rewriter) const {
  // If we still have AssumeNoAlias at this point, just forward the operands.
  // This op is meant to be used in analyses before lowering to improve alias
  // analysis.
  rewriter.replaceOp(op, op.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// AndIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult AndIOpPattern::matchAndRewrite(lsir::AndIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // VCC case short-circuit.
  if (succeeded(lowerVCCBitwise<SAndB32, SAndB64>(op, rewriter, dst, lhs, rhs)))
    return success();

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();

  // At this point, operands are valid - create the appropriate and op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SAndB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SAndB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Move operand to lhs if needed; must happen before any split.
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsKind, rhsKind);
  }

  // Handle the VGPR case; i16 reuses the 32-bit op (upper bits are don't-care).
  if (width <= 32) {
    Value result = createNewVOP<VAndB32>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // 64-bit VGPR: split into two 32-bit ops and recombine.
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);
  Value lo =
      createNewVOP<VAndB32>(rewriter, loc, getElemOr(dstR, 0, dst),
                            getElemOr(lhsR, 0, lhs), getElemOr(rhsR, 0, rhs));
  Value hi =
      createNewVOP<VAndB32>(rewriter, loc, getElemOr(dstR, 1, dst),
                            getElemOr(lhsR, 1, lhs), getElemOr(rhsR, 1, rhs));
  rewriter.replaceOp(op,
                     MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
  return success();
}

//===----------------------------------------------------------------------===//
// KernelOpPattern
//===----------------------------------------------------------------------===//

/// Populate the kernel argument list.
static void addKerArg(SmallVectorImpl<KernelArgAttrInterface> &kerArgs,
                      Type hTy, Type gTy, int32_t size, int32_t align) {
  if (auto pTy = dyn_cast<ptr::PtrType>(hTy)) {
    auto arg = BufferArgAttr::get(gTy.getContext(), AddressSpaceKind::Global,
                                  AccessKind::ReadWrite,
                                  KernelArgumentFlags::None, "", hTy);
    kerArgs.push_back(arg);
    return;
  }
  auto arg = ByValueArgAttr::get(gTy.getContext(), size, align, "", hTy);
  kerArgs.push_back(arg);
}

/// Add hidden kernel arguments.
static void hiddenArgs(SmallVectorImpl<KernelArgAttrInterface> &kerArgs,
                       MLIRContext *ctx) {
  kerArgs.push_back(BlockDimArgAttr::get(ctx, Dim::X));
  kerArgs.push_back(BlockDimArgAttr::get(ctx, Dim::Y));
  kerArgs.push_back(BlockDimArgAttr::get(ctx, Dim::Z));
  kerArgs.push_back(GridDimArgAttr::get(ctx, Dim::X));
  kerArgs.push_back(GridDimArgAttr::get(ctx, Dim::Y));
  kerArgs.push_back(GridDimArgAttr::get(ctx, Dim::Z));
}

LogicalResult
KernelOpPattern::matchAndRewrite(FunctionOpInterface op,
                                 PatternRewriter &rewriter) const {
  if (isa<KernelOp>(op.getOperation()))
    return rewriter.notifyMatchFailure(op, "already a kernel operation");
  auto gFn = dyn_cast<GPUFuncInterface>(op.getOperation());
  // Check this is a GPU kernel function
  if (!gFn || !gFn.isGPUKernel())
    return rewriter.notifyMatchFailure(op, "not a GPU kernel function");

  auto gpuAbiTy = cast<FunctionType>(op.getFunctionType());

  // Get the host ABI information
  auto [type, typeSizes, alignment] = gFn.getHostABI();
  if (!type || type.getNumInputs() != op.getNumArguments() ||
      type.getNumResults() != op.getNumResults()) {
    return rewriter.notifyMatchFailure(op, "invalid host ABI function type");
  }
  if (typeSizes.size() != type.getNumInputs())
    return rewriter.notifyMatchFailure(op, "invalid host ABI size array");
  if (alignment.size() != type.getNumInputs())
    return rewriter.notifyMatchFailure(op, "invalid host ABI alignment array");
  if (!llvm::all_of(gpuAbiTy.getInputs(), llvm::IsaPred<SGPRType, SGPRType>))
    return rewriter.notifyMatchFailure(op, "expected all inputs to be SGPRs");

  // Set Metadata attributes
  int32_t smemSize = gFn.getSharedMemorySize();
  SmallVector<KernelArgAttrInterface> kerArgs;
  for (auto [hTy, dTy, sz, align] :
       llvm::zip(cast<FunctionType>(op.getFunctionType()).getInputs(),
                 type.getInputs(), typeSizes, alignment)) {
    addKerArg(kerArgs, hTy, dTy, sz, align);
  }
  hiddenArgs(kerArgs, op.getContext());

  // Create the KernelOp
  auto kOp = amdgcn::KernelOp::create(
      rewriter, op.getLoc(), op.getName(), kerArgs, smemSize,
      /*private_memory_size=*/0, /*enable_private_segment_buffer=*/false,
      /*enable_dispatch_ptr=*/false,
      /*enable_kernarg_segment_ptr=*/!kerArgs.empty());
  rewriter.inlineRegionBefore(op.getFunctionBody(), kOp.getBodyRegion(),
                              kOp.getBodyRegion().end());
  Block *entry = &kOp.getBodyRegion().front();
  // Replace arguments with LoadArgOps
  rewriter.setInsertionPointToStart(entry);
  int64_t numArgs = op.getNumArguments();
  for (auto [i, arg] : llvm::enumerate(llvm::reverse(entry->getArguments()))) {
    int64_t idx = numArgs - i - 1;
    assert(idx >= 0 && idx < numArgs && "invalid argument index");
    if (arg.use_empty())
      continue;
    Value rA = LoadArgOp::create(rewriter, op.getLoc(), arg.getType(), idx);
    rewriter.replaceAllUsesWith(arg, rA);
  }
  entry->eraseArguments(0, numArgs);

  // Replace the function op with the kernel op
  rewriter.replaceOp(op, kOp);
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOpPattern
//===----------------------------------------------------------------------===//

LogicalResult LoadOpPattern::matchAndRewrite(lsir::LoadOp op,
                                             PatternRewriter &rewriter) const {
  auto memSpace = cast<amdgcn::AddressSpaceAttr>(op.getMemorySpace());
  if (!memSpace) {
    return rewriter.notifyMatchFailure(
        op, "expected AMDGCN address space attribute for load operation");
  }

  // Check dependencies
  if (op.getDependencies().size() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "load operation with dependencies are not supported by this pattern");
  }
  if (!op.getOutDependency().use_empty()) {
    return rewriter.notifyMatchFailure(
        op, "can't handle load operation with out dependency in this pattern");
  }

  // Check memory space
  AddressSpaceKind space = memSpace.getSpace();
  if (!isAddressSpaceOf(space,
                        {AddressSpaceKind::Global, AddressSpaceKind::Local})) {
    return rewriter.notifyMatchFailure(
        op,
        "only global and local memory spaces are supported by this pattern");
  }

  // Get constant offset
  int32_t off = 0;
  if (std::optional<int32_t> constOff =
          ValueOrI32::getConstant(op.getConstOffset())) {
    off = *constOff;
  } else {
    return rewriter.notifyMatchFailure(
        op, "only constant offsets are supported by this pattern");
  }

  Location loc = op.getLoc();
  TypedValue<RegisterTypeInterface> dst = op.getDst();
  TypedValue<RegisterTypeInterface> addr = op.getAddr();
  Value offset = op.getOffset();
  RegisterTypeInterface addrTy = addr.getType();
  RegisterTypeInterface resTy = dst.getType();
  Value result;

  // Check if the offset is constant and add it to the constant offset.
  if (std::optional<int32_t> constOff = ValueOrI32::getConstant(offset)) {
    off += *constOff;
    offset = nullptr;
  }

  // Number of 32-bit words to load
  int16_t numWords = resTy.getAsRange().size();
  if (space == AddressSpaceKind::Local) {
    if (!isVGPR(addrTy, 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected VGPR address for load from shared memory space");
    }
    if (offset) {
      return rewriter.notifyMatchFailure(op,
                                         "only constant offsets are supported "
                                         "for load from shared memory space");
    }
    offset = getI32Constant(rewriter, loc, off);
    // Trailing optionals (gds, fence_token): no GDS flag, no fence token.
    UnitAttr noGds = {};
    Value noFenceToken = {};
    switch (numWords) {
    case 1:
      result = DsReadB32::create(rewriter, loc, dst, addr, offset, noGds,
                                 noFenceToken)
                   .getDestRes();
      break;
    case 2:
      result = DsReadB64::create(rewriter, loc, dst, addr, offset, noGds,
                                 noFenceToken)
                   .getDestRes();
      break;
    case 3:
      result = DsReadB96::create(rewriter, loc, dst, addr, offset, noGds,
                                 noFenceToken)
                   .getDestRes();
      break;
    case 4:
      result = DsReadB128::create(rewriter, loc, dst, addr, offset, noGds,
                                  noFenceToken)
                   .getDestRes();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for load from shared memory space");
    }
    rewriter.replaceAllUsesWith(op.getDstRes(), result);
    rewriter.eraseOp(op);
    return success();
  }
  Value cOff = getI32Constant(rewriter, loc, off);

  // Handle a SMEM load.
  bool addrIsSGPR = isSGPR(addrTy, 2);
  if (addrIsSGPR && (!offset || isSGPR(offset.getType(), 1))) {
    if (offset) {
      return rewriter.notifyMatchFailure(op, "nyi: SGPR offset");
    }
    switch (numWords) {
    case 1:
      result = SLoadDword::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 2:
      result = SLoadDwordx2::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 4:
      result = SLoadDwordx4::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 8:
      result = SLoadDwordx8::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 16:
      result = SLoadDwordx16::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for load from shared memory space");
    }
    rewriter.replaceAllUsesWith(op.getDstRes(), result);
    rewriter.eraseOp(op);
    return success();
  }

  // Handle a VMEM load
  bool addrIsVGPR = isVGPR(addrTy, 2);
  if (!addrIsVGPR && !addrIsSGPR) {
    return rewriter.notifyMatchFailure(
        op, "expected VGPR or SGPR address for load from global memory space");
  }
  if (addrIsVGPR && offset) {
    return rewriter.notifyMatchFailure(
        op, "expected no offset or SGPR address for load");
  }
  if (addrIsSGPR && offset && !isVGPR(offset.getType(), 1)) {
    return rewriter.notifyMatchFailure(
        op, "expected VGPR offset for load from global memory space");
  }
  switch (numWords) {
  case 1:
    result = GlobalLoadDword::create(rewriter, loc, dst, addr, offset, cOff)
                 .getDestRes();
    break;
  case 2:
    result = GlobalLoadDwordx2::create(rewriter, loc, dst, addr, offset, cOff)
                 .getDestRes();
    break;
  case 3:
    result = GlobalLoadDwordx3::create(rewriter, loc, dst, addr, offset, cOff)
                 .getDestRes();
    break;
  case 4:
    result = GlobalLoadDwordx4::create(rewriter, loc, dst, addr, offset, cOff)
                 .getDestRes();
    break;
  default:
    return rewriter.notifyMatchFailure(
        op, "unsupported number of words for load from global memory space");
  }
  rewriter.replaceAllUsesWith(op.getDstRes(), result);
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// MulIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult MulIOpPattern::matchAndRewrite(lsir::MulIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // If dest is SGPR but a source is VGPR (mixed uniform/dynamic), use VGPR
  // path and copy result back. Override kind/oTy for validation.
  bool sgprDestVgprSrc = false;
  handleMixedSgprVgprDest(kind, oTy, lhs, rhs, sgprDestVgprSrc,
                          op.getContext());

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();
  // Maybe split operands if they are register ranges
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsR, rhsR);
    std::swap(lhsKind, rhsKind);
  }

  // If lhs is a constant that doesn't fit in 6 bits, move it to a VGPR.
  // TODO: this is just a very quick and approximate fix, we should have a
  // general solution.
  if (kind == OperandKind::VGPR && lhsKind == OperandKind::IntImm) {
    APInt constVal;
    if (matchPattern(lhs, m_ConstantInt(&constVal)) &&
        !constVal.isSignedIntN(6)) {
      Value vgpr = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
      lhs = VMovB32::create(rewriter, loc, vgpr, lhs).getDst0Res();
      lhsKind = OperandKind::VGPR;
    }
  }

  // At this point, operands are valid - create the appropriate mul op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = SMulI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
      rewriter.replaceOp(op, result);
      return success();
    }

    // 64-bit SGPR multiplication
    Value lLo = getElemOr(lhsR, 0, lhs);
    Value lHi = getElemOr(lhsR, 1, lhs);
    Value rLo = getElemOr(rhsR, 0, rhs);
    Value rHi = getElemOr(rhsR, 1, rhs);
    Value dLo = getElemOr(dstR, 0, dst);
    Value dHi = getElemOr(dstR, 1, dst);
    Value t0 = createAllocation(rewriter, loc, getSGPR(getCtx(rewriter)));
    Value t1 = createAllocation(rewriter, loc, getSGPR(getCtx(rewriter)));

    dHi = SMulI32::create(rewriter, loc, dHi, rLo, lHi).getDst0Res();
    t0 = SMulHiU32::create(rewriter, loc, t0, rLo, lLo).getDst0Res();
    t1 = SMulI32::create(rewriter, loc, t1, rHi, lLo).getDst0Res();
    dHi = createSOP2Out2In2<SAddI32>(rewriter, loc, dHi, t0, dHi);
    dLo = SMulI32::create(rewriter, loc, dLo, rLo, lLo).getDst0Res();
    dHi = createSOP2Out2In2<SAddI32>(rewriter, loc, dHi, dHi, t1);

    // Combine low and high parts
    Value result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {dLo, dHi});
    rewriter.replaceOp(op, result);
    return success();
  }

  // Allocate VGPR temp if we're writing to SGPR from VGPR path.
  Value actualDst = dst;
  if (sgprDestVgprSrc) {
    int16_t rangeSize = oTy.getAsRange().size();
    actualDst = createAllocation(rewriter, loc,
                                 getVGPR(rewriter.getContext(), rangeSize));
  }

  // Handle the VGPR case
  Value result;
  if (width <= 16) {
    result = createNewVOP<VMulLoU16>(rewriter, loc, actualDst, lhs, rhs);
  } else if (width <= 32) {
    if (lhsKind == OperandKind::IntImm) {
      APInt constVal;
      if (matchPattern(lhs, m_ConstantInt(&constVal))) {
        Value vgpr = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
        lhs = VMovB32::create(rewriter, loc, vgpr, lhs).getDst0Res();
        lhsKind = OperandKind::VGPR;
      }
    }
    result = VMulLoU32::create(rewriter, loc, actualDst, lhs, rhs).getDst0Res();
  } else {
    // 64-bit VGPR multiplication
    Value lLo = getElemOr(lhsR, 0, lhs);
    Value lHi = getElemOr(lhsR, 1, lhs);
    Value rLo = getElemOr(rhsR, 0, rhs);
    Value rHi = getElemOr(rhsR, 1, rhs);

    // Allocate temporaries
    Value t0 = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
    Value t1 = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
    Value carry = createAllocation(rewriter, loc, getSGPR(getCtx(rewriter), 2));
    t0 = VMulLoU32::create(rewriter, loc, t0, rHi, lLo).getDst0Res();
    t1 = VMulLoU32::create(rewriter, loc, t1, rLo, lHi).getDst0Res();
    Value zero = getI32Constant(rewriter, loc, 0);
    ValueRange dT0 = splitRange(
        rewriter, loc,
        VMadU64U32::create(rewriter, loc, actualDst, carry, rLo, lLo, zero)
            .getDst0Res());
    Value t3 =
        VAdd3U32::create(rewriter, loc, dT0[1], dT0[1], t1, t0).getDst0Res();
    result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {dT0[0], t3});
  }

  // Copy VGPR result back to SGPR destination.
  result = finishMixedSgprVgprDest(rewriter, loc, dst, result, sgprDestVgprSrc);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// MulHiSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
MulHiSIOpPattern::matchAndRewrite(lsir::MulHiSIOp op,
                                  PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32}, {32})))
    return failure();

  Location loc = op.getLoc();

  if (kind == OperandKind::SGPR) {
    Value result = SMulHiI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }

  // VOP3 does not support literal constants -- move to VGPR first.
  auto movLargeImm = [&](Value &v, OperandKind vKind) {
    if (vKind != OperandKind::IntImm)
      return;
    APInt constVal;
    if (matchPattern(v, m_ConstantInt(&constVal)) &&
        !constVal.isSignedIntN(6)) {
      Value vgpr = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
      v = VMovB32::create(rewriter, loc, vgpr, v).getDst0Res();
    }
  };
  movLargeImm(lhs, lhsKind);
  movLargeImm(rhs, rhsKind);

  Value result = VMulHiI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// OrIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult OrIOpPattern::matchAndRewrite(lsir::OrIOp op,
                                            PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // VCC case short-circuit.
  if (succeeded(lowerVCCBitwise<SOrB32, SOrB64>(op, rewriter, dst, lhs, rhs)))
    return success();

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();

  // At this point, operands are valid - create the appropriate or op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SOrB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SOrB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Move operand to lhs if needed; must happen before any split.
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsKind, rhsKind);
  }

  // Handle the VGPR case; i16 reuses the 32-bit op (upper bits are don't-care).
  if (width <= 32) {
    Value result = createNewVOP<VOrB32>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // 64-bit VGPR: split into two 32-bit ops and recombine.
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);
  Value lo =
      createNewVOP<VOrB32>(rewriter, loc, getElemOr(dstR, 0, dst),
                           getElemOr(lhsR, 0, lhs), getElemOr(rhsR, 0, rhs));
  Value hi =
      createNewVOP<VOrB32>(rewriter, loc, getElemOr(dstR, 1, dst),
                           getElemOr(lhsR, 1, lhs), getElemOr(rhsR, 1, rhs));
  rewriter.replaceOp(op,
                     MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
  return success();
}

//===----------------------------------------------------------------------===//
// XOrIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult XOrIOpPattern::matchAndRewrite(lsir::XOrIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // VCC case short-circuit.
  if (succeeded(lowerVCCBitwise<SXorB32, SXorB64>(op, rewriter, dst, lhs, rhs)))
    return success();

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {32})))
    return failure();

  Location loc = op.getLoc();

  // At this point, operands are valid - create the appropriate xor op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SXorB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SXorB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsKind, rhsKind);
  }

  // Handle the VGPR case
  Value result = createNewVOP<VXorB32>(rewriter, loc, dst, lhs, rhs);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// MovOpPattern
//===----------------------------------------------------------------------===//

LogicalResult MovOpPattern::matchAndRewrite(lsir::MovOp op,
                                            PatternRewriter &rewriter) const {
  // Only handle the constant case.
  if (!matchPattern(op.getValue(), m_Constant()))
    return rewriter.notifyMatchFailure(op, "only constant mov is supported");

  OperandKind kind = getOperandKind(op.getDst().getType());
  Value res;
  switch (kind) {
  case OperandKind::VGPR:
    res = VMovB32::create(rewriter, op.getLoc(), op.getDst(), op.getValue())
              .getDst0Res();
    break;
  case OperandKind::SGPR:
    res = SMovB32::create(rewriter, op.getLoc(), op.getDst(), op.getValue())
              .getDst0Res();
    break;
  default:
    return rewriter.notifyMatchFailure(op, "unsupported mov register operand");
  }
  rewriter.replaceOp(op, res);
  return success();
}

//===----------------------------------------------------------------------===//
// RegCastOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
RegCastOpPattern::matchAndRewrite(lsir::RegCastOp op,
                                  PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  OperandKind srcKind = getOperandKind(op.getSrc().getType());
  OperandKind tgtKind = getOperandKind(op.getType());
  if (srcKind != OperandKind::SGPR || tgtKind != OperandKind::VGPR) {
    return rewriter.notifyMatchFailure(
        op, "Can only handle SGPR to VGPR conversion");
  }
  if (op.getSrc().getType().getAsRange().size() != 1 ||
      op.getType().getAsRange().size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Can only handle single word conversion conversion");
  }

  Value res = VMovB32::create(rewriter, loc,
                              createAllocation(rewriter, loc, op.getType()),
                              op.getSrc())
                  .getDst0Res();
  rewriter.replaceOp(op, res);
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpPattern::matchAndRewrite(func::ReturnOp op,
                                 PatternRewriter &rewriter) const {
  if (op->getParentOfType<KernelOp>() == nullptr)
    return failure();
  rewriter.replaceOpWithNewOp<EndKernelOp>(op);
  return success();
}

//===----------------------------------------------------------------------===//
// ShLIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShLIOpPattern::matchAndRewrite(lsir::ShLIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // If dest is SGPR but source is VGPR (mixed uniform/dynamic), use VGPR
  // path and copy result back. Override kind for validation.
  bool sgprDestVgprSrc = false;
  OperandKind srcKind = getOperandKind(lhs.getType());
  if (kind == OperandKind::SGPR && srcKind == OperandKind::VGPR) {
    sgprDestVgprSrc = true;
    kind = OperandKind::VGPR;
    oTy = cast<RegisterTypeInterface>(getVGPR(rewriter.getContext(), 1));
  }

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();
  Location loc = op.getLoc();

  // Handle the SGPR case (only if all operands fit SGPR path).
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SLshlB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SLshlB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Allocate VGPR temp if we're writing to SGPR from VGPR path.
  Value actualDst = dst;
  if (sgprDestVgprSrc) {
    actualDst =
        createAllocation(rewriter, loc, getVGPR(rewriter.getContext(), 1));
  }

  // Handle the VGPR case
  Value result;
  if (width == 16) {
    // NOTE: Operands are reversed
    result = createNewVOP<VLshlrevB16>(rewriter, loc, actualDst, rhs, lhs);
  } else if (width == 32) {
    // NOTE: Operands are reversed
    result = createNewVOP<VLshlrevB32>(rewriter, loc, actualDst, rhs, lhs);
  } else {
    // NOTE: Operands are reversed
    result =
        VLshlrevB64::create(rewriter, loc, actualDst, rhs, lhs).getDst0Res();
  }

  // Copy VGPR result back to SGPR destination.
  if (sgprDestVgprSrc) {
    result = lsir::CopyOp::create(rewriter, loc, dst, result).getTargetRes();
  }
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ShRSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShRSIOpPattern::matchAndRewrite(lsir::ShRSIOp op,
                                              PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();
  Location loc = op.getLoc();

  // Handle the SGPR case
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SAshrI32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SAshrI64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width == 16) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VAshrrevI16>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width == 32) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VAshrrevI32>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  // NOTE: Operands are reversed
  Value result = VAshrrevI64::create(rewriter, loc, dst, rhs, lhs).getDst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ShRUIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShRUIOpPattern::matchAndRewrite(lsir::ShRUIOp op,
                                              PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();
  Location loc = op.getLoc();

  // Handle the SGPR case
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SLshrB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SLshrB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width == 16) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VLshrrevB16>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width == 32) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VLshrrevB32>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  // NOTE: Operands are reversed
  Value result = VLshrrevB64::create(rewriter, loc, dst, rhs, lhs).getDst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ExtSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ExtSIOpPattern::matchAndRewrite(lsir::ExtSIOp op,
                                              PatternRewriter &rewriter) const {
  unsigned srcWidth = op.getSrcType().getWidth();
  unsigned tgtWidth = op.getTgtType().getWidth();
  if (srcWidth != 32 || tgtWidth != 64)
    return rewriter.notifyMatchFailure(
        op, "only i32 to i64 and u32 to u64 sign extension is supported");

  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value value = op.getValue();
  OperandKind kind = getOperandKind(oTy);

  if (!isAMDReg(oTy))
    return rewriter.notifyMatchFailure(op, "dst must be AMDGCN register type");
  if (kind != OperandKind::SGPR && kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "only SGPR and VGPR are supported");
  if (oTy.getAsRange().size() != 2)
    return rewriter.notifyMatchFailure(
        op, "dst must be a 2-register range for i64");

  Location loc = op.getLoc();
  ValueRange dstR = splitRange(rewriter, loc, dst);
  if ((int)dstR.size() != 2)
    return rewriter.notifyMatchFailure(op, "dst must be a splittable range");

  Value dstLo = dstR[0];
  Value dstHi = dstR[1];

  // Sign extension: lo = value, hi = sign_extend(value) = shrsi(value, 31)
  Value shiftAmount = getI32Constant(rewriter, loc, 31);
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());
  Value lo = lsir::CopyOp::create(rewriter, loc, dstLo, value).getTargetRes();
  Value hi = lsir::ShRSIOp::create(rewriter, loc, i32Semantics, dstHi, value,
                                   shiftAmount)
                 .getDstRes();

  Value result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi});
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ExtUIOpPattern
//===----------------------------------------------------------------------===//

/// Combine lo and hi values into a 64-bit register range.
static Value combineLoHiToI64(PatternRewriter &rewriter, Location loc,
                              RegisterTypeInterface oTy, Value lo, Value hi) {
  return MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi});
}

/// Zero-extend a sub-32-bit integer into a 32-bit register by masking the live
/// bits.
static Value zeroExtendToI32(PatternRewriter &rewriter, Location loc,
                             OperandKind kind, Value dst, Value value,
                             int srcWidth) {
  assert(srcWidth < 32 && "srcWidth must be < 32");
  int32_t mask = static_cast<int32_t>((1u << srcWidth) - 1u);
  Value maskVal = getI32Constant(rewriter, loc, mask);
  if (kind == OperandKind::SGPR)
    return createSOP2Out2In2<SAndB32>(rewriter, loc, dst, value, maskVal);
  // VOP does not support non-inline literal constants; materialize the mask
  // in a VGPR so v_and_b32 can use a register operand.
  bool isInline = mask >= -16 && mask <= 64;
  if (!isInline) {
    Value maskVgpr =
        createAllocation(rewriter, loc, getVGPR(rewriter.getContext()));
    maskVal = VMovB32::create(rewriter, loc, maskVgpr, maskVal).getDst0Res();
  }
  return createNewVOP<VAndB32>(rewriter, loc, dst, value, maskVal);
}

LogicalResult ExtUIOpPattern::matchAndRewrite(lsir::ExtUIOp op,
                                              PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());

  RegisterTypeInterface oTy = op.getDst().getType();
  int16_t rangeSize = oTy.getAsRange().size();
  Value dst = op.getDst();
  Value value = op.getValue();
  OperandKind kind = getOperandKind(oTy);

  if (!isAMDReg(oTy))
    return rewriter.notifyMatchFailure(op, "dst must be AMDGCN register type");
  if (kind != OperandKind::SGPR && kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "only SGPR and VGPR are supported");

  Location loc = op.getLoc();

  // narrow -> i32: mask off high bits into a single destination register.
  if (tgtWidth == 32 && (srcWidth == 1 || srcWidth == 8 || srcWidth == 16)) {
    if (rangeSize != 1)
      return rewriter.notifyMatchFailure(
          op, "dst must be a single register for i32");
    Value result = zeroExtendToI32(rewriter, loc, kind, dst, value, srcWidth);
    rewriter.replaceOp(op, result);
    return success();
  }

  // i32 -> i64: copy value into lo, zero into hi.
  if (srcWidth == 32 && tgtWidth == 64) {
    if (rangeSize != 2)
      return rewriter.notifyMatchFailure(
          op, "dst must be a 2-register range for i64");
    ValueRange dstR = splitRange(rewriter, loc, dst);
    if ((int)dstR.size() != 2)
      return rewriter.notifyMatchFailure(op, "dst must be a splittable range");
    Value hi = lsir::MovOp::create(rewriter, loc, dstR[1],
                                   getI32Constant(rewriter, loc, 0))
                   .getDstRes();
    Value lo =
        lsir::CopyOp::create(rewriter, loc, dstR[0], value).getTargetRes();
    rewriter.replaceOp(op, combineLoHiToI64(rewriter, loc, oTy, lo, hi));
    return success();
  }

  // narrow -> i64: mask into lo, zero into hi.
  if (tgtWidth == 64 && (srcWidth == 1 || srcWidth == 8 || srcWidth == 16)) {
    if (rangeSize != 2)
      return rewriter.notifyMatchFailure(
          op, "dst must be a 2-register range for i64");
    ValueRange dstR = splitRange(rewriter, loc, dst);
    if ((int)dstR.size() != 2)
      return rewriter.notifyMatchFailure(op, "dst must be a splittable range");
    Value lo = zeroExtendToI32(rewriter, loc, kind, dstR[0], value, srcWidth);
    Value hi = lsir::MovOp::create(rewriter, loc, dstR[1],
                                   getI32Constant(rewriter, loc, 0))
                   .getDstRes();
    rewriter.replaceOp(op, combineLoHiToI64(rewriter, loc, oTy, lo, hi));
    return success();
  }

  return rewriter.notifyMatchFailure(op, "unsupported extui width combination");
}

//===----------------------------------------------------------------------===//
// TruncIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
TruncIOpPattern::matchAndRewrite(lsir::TruncIOp op,
                                 PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value value = op.getValue();
  OperandKind kind = getOperandKind(oTy);
  Location loc = op.getLoc();

  if (!isAMDReg(oTy))
    return rewriter.notifyMatchFailure(op,
                                       "dst must be an AMDGCN register type");
  if (kind != OperandKind::SGPR && kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "only SGPR and VGPR are supported");

  // i64 -> i32: result is the low register of the source pair.
  if (srcWidth == 64 && tgtWidth == 32) {
    ValueRange valR = splitRange(rewriter, loc, value);
    if ((int)valR.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "source must be a 2-register range for i64");
    Value result =
        lsir::CopyOp::create(rewriter, loc, dst, valR[0]).getTargetRes();
    rewriter.replaceOp(op, result);
    return success();
  }

  // i32 -> i16: low 16 bits live in the low half; a plain copy suffices.
  if (srcWidth == 32 && tgtWidth == 16) {
    Value result =
        lsir::CopyOp::create(rewriter, loc, dst, value).getTargetRes();
    rewriter.replaceOp(op, result);
    return success();
  }

  // i32 -> i8 / i32 -> i1: mask off the high bits (helper handles SGPR/VGPR
  // dispatch and non-inline mask materialization).
  if (srcWidth == 32 && (tgtWidth == 8 || tgtWidth == 1)) {
    Value result = zeroExtendToI32(rewriter, loc, kind, dst, value, tgtWidth);
    rewriter.replaceOp(op, result);
    return success();
  }

  return rewriter.notifyMatchFailure(op,
                                     "unsupported trunci width combination");
}

//===----------------------------------------------------------------------===//
// ExtFOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ExtFOpPattern::matchAndRewrite(lsir::ExtFOp op,
                                             PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());
  Value dst = op.getDst();
  Value value = op.getValue();
  Location loc = op.getLoc();

  if (getOperandKind(dst.getType()) != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "extf dst must be a VGPR");

  if (srcWidth == 16 && tgtWidth == 32) {
    if (cast<RegisterTypeInterface>(dst.getType()).getAsRange().size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "f32 dst must be a single register");
    Value result = VCvtF32F16::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  if (srcWidth == 32 && tgtWidth == 64) {
    if (cast<RegisterTypeInterface>(dst.getType()).getAsRange().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "f64 dst must be a 2-register range");
    Value result = VCvtF64F32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "unsupported extf width combination");
}

//===----------------------------------------------------------------------===//
// TruncFOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
TruncFOpPattern::matchAndRewrite(lsir::TruncFOp op,
                                 PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());
  Value dst = op.getDst();
  Value value = op.getValue();
  Location loc = op.getLoc();

  if (getOperandKind(dst.getType()) != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "truncf dst must be a VGPR");

  if (srcWidth == 32 && tgtWidth == 16) {
    if (cast<RegisterTypeInterface>(dst.getType()).getAsRange().size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "f16 dst must be a single register");
    Value result = VCvtF16F32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  if (srcWidth == 64 && tgtWidth == 32) {
    if (cast<RegisterTypeInterface>(value.getType()).getAsRange().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "f64 src must be a 2-register range");
    Value result = VCvtF32F64::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op,
                                     "unsupported truncf width combination");
}

//===----------------------------------------------------------------------===//
// SIToFPOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
SIToFPOpPattern::matchAndRewrite(lsir::SIToFPOp op,
                                 PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());
  Value dst = op.getDst();
  Value value = op.getValue();
  Location loc = op.getLoc();

  if (getOperandKind(dst.getType()) != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "sitofp dst must be a VGPR");

  if (srcWidth == 32 && tgtWidth == 32) {
    Value result = VCvtF32I32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  if (srcWidth == 32 && tgtWidth == 64) {
    if (cast<RegisterTypeInterface>(dst.getType()).getAsRange().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "f64 dst must be a 2-register range");
    Value result = VCvtF64I32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op,
                                     "unsupported sitofp width combination");
}

//===----------------------------------------------------------------------===//
// UIToFPOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
UIToFPOpPattern::matchAndRewrite(lsir::UIToFPOp op,
                                 PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());
  Value dst = op.getDst();
  Value value = op.getValue();
  Location loc = op.getLoc();

  if (getOperandKind(dst.getType()) != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "uitofp dst must be a VGPR");

  if (srcWidth == 32 && tgtWidth == 32) {
    Value result = VCvtF32U32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  if (srcWidth == 32 && tgtWidth == 64) {
    if (cast<RegisterTypeInterface>(dst.getType()).getAsRange().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "f64 dst must be a 2-register range");
    Value result = VCvtF64U32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op,
                                     "unsupported uitofp width combination");
}

//===----------------------------------------------------------------------===//
// FPToSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
FPToSIOpPattern::matchAndRewrite(lsir::FPToSIOp op,
                                 PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());
  Value dst = op.getDst();
  Value value = op.getValue();
  Location loc = op.getLoc();

  if (getOperandKind(dst.getType()) != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "fptosi dst must be a VGPR");

  if (srcWidth == 32 && tgtWidth == 32) {
    Value result = VCvtI32F32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  if (srcWidth == 64 && tgtWidth == 32) {
    if (cast<RegisterTypeInterface>(value.getType()).getAsRange().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "f64 src must be a 2-register range");
    Value result = VCvtI32F64::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op,
                                     "unsupported fptosi width combination");
}

//===----------------------------------------------------------------------===//
// FPToUIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
FPToUIOpPattern::matchAndRewrite(lsir::FPToUIOp op,
                                 PatternRewriter &rewriter) const {
  int srcWidth = static_cast<int>(op.getSrcType().getWidth());
  int tgtWidth = static_cast<int>(op.getTgtType().getWidth());
  Value dst = op.getDst();
  Value value = op.getValue();
  Location loc = op.getLoc();

  if (getOperandKind(dst.getType()) != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "fptoui dst must be a VGPR");

  if (srcWidth == 32 && tgtWidth == 32) {
    Value result = VCvtU32F32::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  if (srcWidth == 64 && tgtWidth == 32) {
    if (cast<RegisterTypeInterface>(value.getType()).getAsRange().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "f64 src must be a 2-register range");
    Value result = VCvtU32F64::create(rewriter, loc, dst, value).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op,
                                     "unsupported fptoui width combination");
}

//===----------------------------------------------------------------------===//
// StoreOpPattern
//===----------------------------------------------------------------------===//

LogicalResult StoreOpPattern::matchAndRewrite(lsir::StoreOp op,
                                              PatternRewriter &rewriter) const {
  auto memSpace = dyn_cast<amdgcn::AddressSpaceAttr>(op.getMemorySpace());
  if (!memSpace) {
    return rewriter.notifyMatchFailure(
        op, "expected AMDGCN address space attribute for store operation");
  }

  // Check dependencies
  if (op.getDependencies().size() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "store operation with dependencies are not supported by this pattern");
  }
  if (!op.getOutDependency().use_empty()) {
    return rewriter.notifyMatchFailure(
        op, "can't handle load operation with out dependency in this pattern");
  }

  // Check memory space
  AddressSpaceKind space = memSpace.getSpace();
  if (!isAddressSpaceOf(space,
                        {AddressSpaceKind::Global, AddressSpaceKind::Local})) {
    return rewriter.notifyMatchFailure(
        op,
        "only global and local memory spaces are supported by this pattern");
  }

  // Get constant offset
  int32_t off = 0;
  if (std::optional<int32_t> constOff =
          ValueOrI32::getConstant(op.getConstOffset())) {
    off = *constOff;
  } else {
    return rewriter.notifyMatchFailure(
        op, "only constant offsets are supported by this pattern");
  }

  Location loc = op.getLoc();
  TypedValue<RegisterTypeInterface> data = op.getValue();
  Value addr = op.getAddr();
  Value offset = op.getOffset();
  RegisterTypeInterface dataTy = data.getType();
  Type addrTy = addr.getType();

  // Check if the offset is constant and add it to the constant offset.
  if (std::optional<int32_t> constOff = ValueOrI32::getConstant(offset)) {
    off += *constOff;
    offset = nullptr;
  }

  // Number of 32-bit words to store
  int16_t numWords = dataTy.getAsRange().size();

  // Handle local memory store (DS)
  if (space == AddressSpaceKind::Local) {
    auto vgprAddrTy = dyn_cast<VGPRType>(addrTy);
    if (!vgprAddrTy) {
      return rewriter.notifyMatchFailure(
          op, "expected VGPR address for store to shared memory space");
    }
    if (offset) {
      return rewriter.notifyMatchFailure(op,
                                         "only constant offsets are supported "
                                         "for store to shared memory space");
    }
    offset = getI32Constant(rewriter, loc, off);

    // Convert data to VGPRType if needed
    Value dataRange = data;

    switch (numWords) {
    case 1:
      DsWriteB32::create(rewriter, loc, addr, dataRange, offset);
      break;
    case 2:
      DsWriteB64::create(rewriter, loc, addr, dataRange, offset);
      break;
    case 3:
      DsWriteB96::create(rewriter, loc, addr, dataRange, offset);
      break;
    case 4:
      DsWriteB128::create(rewriter, loc, addr, dataRange, offset);
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for store to shared memory space");
    }
    rewriter.eraseOp(op);
    return success();
  }

  // Handle global memory store
  auto addrRegTy = dyn_cast<RegisterTypeInterface>(addrTy);
  if (!addrRegTy) {
    return rewriter.notifyMatchFailure(
        op, "expected register type address for store to global memory space");
  }

  bool addrIsSGPR = isSGPR(addrRegTy, 2);
  bool addrIsVGPR = isVGPR(addrRegTy, 2);

  // Handle SMEM store (SGPR address, SGPR data)
  if (addrIsSGPR && isSGPR(dataTy, -1)) {
    if (offset && !isSGPR(offset.getType(), 1)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected SGPR offset for SMEM store");
    }
    if (offset) {
      return rewriter.notifyMatchFailure(op, "nyi: SGPR offset for SMEM store");
    }

    switch (numWords) {
    case 1:
      SStoreDword::create(rewriter, loc, data, addr, nullptr,
                          getI32Constant(rewriter, loc, off));
      break;
    case 2:
      SStoreDwordx2::create(rewriter, loc, data, addr, nullptr,
                            getI32Constant(rewriter, loc, off));
      break;
    case 4:
      SStoreDwordx4::create(rewriter, loc, data, addr, nullptr,
                            getI32Constant(rewriter, loc, off));
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for SMEM store");
    }
    rewriter.eraseOp(op);
    return success();
  }

  // Handle VMEM store (global_store)
  if (!addrIsVGPR && !addrIsSGPR) {
    return rewriter.notifyMatchFailure(
        op, "expected VGPR or SGPR address for store to global memory space");
  }
  if (addrIsVGPR && offset) {
    return rewriter.notifyMatchFailure(
        op, "expected no offset with VGPR address for global store");
  }
  if (addrIsSGPR && offset && !isVGPR(offset.getType(), 1)) {
    return rewriter.notifyMatchFailure(op,
                                       "expected VGPR offset for store to "
                                       "global memory space with SGPR address");
  }
  Value cOff = getI32Constant(rewriter, loc, off);
  switch (numWords) {
  case 1:
    GlobalStoreDword::create(rewriter, loc, data, addr, offset, cOff);
    break;
  case 2:
    GlobalStoreDwordx2::create(rewriter, loc, data, addr, offset, cOff);
    break;
  case 3:
    GlobalStoreDwordx3::create(rewriter, loc, data, addr, offset, cOff);
    break;
  case 4:
    GlobalStoreDwordx4::create(rewriter, loc, data, addr, offset, cOff);
    break;
  default:
    return rewriter.notifyMatchFailure(
        op, "unsupported number of words for store to global memory space");
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// SubIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult SubIOpPattern::matchAndRewrite(lsir::SubIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {16, 32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();
  // Maybe split operands if they are register ranges
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsR, rhsR);
    std::swap(lhsKind, rhsKind);
  }

  // At this point, operands are valid - create the appropriate add op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SSubU32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value lo = createSOP2Out2In2<SSubU32>(
        rewriter, loc, getElemOr(dstR, 0, dst), getElemOr(lhsR, 0, lhs),
        getElemOr(rhsR, 0, rhs));
    Value hi = createSOP2Out2In3<SSubbU32>(
        rewriter, loc, getElemOr(dstR, 1, dst), getElemOr(lhsR, 1, lhs),
        getElemOr(rhsR, 1, rhs));
    rewriter.replaceOp(
        op, MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
    return success();
  }

  // Handle the VGPR case
  if (width <= 16) {
    Value result = createNewVOP<VSubU16>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width <= 32) {
    Value result = createNewVOP<VSubU32>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // 64-bit VGPR sub
  Value carry = createAllocation(
      rewriter, loc, rewriter.getType<SGPRType>(RegisterRange(Register(), 2)));

  Value lo = VSubCoU32::create(rewriter, loc, getElemOr(dstR, 0, dst), carry,
                               getElemOr(lhsR, 0, lhs), getElemOr(rhsR, 0, rhs))
                 .getDst0Res();
  Value hi = VSubbCoU32::create(rewriter, loc, getElemOr(dstR, 1, dst), carry,
                                getElemOr(lhsR, 1, lhs),
                                getElemOr(rhsR, 1, rhs), carry)
                 .getDst0Res();
  rewriter.replaceOp(op,
                     MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

/// Create ptr + offset with sign-extension to i64.
static Value addPtrWithSignExtend(PatternRewriter &rewriter, Location loc,
                                  Value ptr, Value offset,
                                  RegisterTypeInterface resultTy,
                                  RegisterTypeInterface extDstTy) {
  auto i64Semantics = TypeAttr::get(rewriter.getI64Type());
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());
  Value extDst = createAllocation(rewriter, loc, extDstTy);
  Value extVal = lsir::ExtSIOp::create(rewriter, loc, i64Semantics,
                                       i32Semantics, extDst, offset)
                     .getDstRes();
  Value addDst = createAllocation(rewriter, loc, resultTy);
  return lsir::AddIOp::create(rewriter, loc, i64Semantics, addDst, ptr, extVal)
      .getDstRes();
}

/// Create ptr + offset (no extension, i32).
static Value addPtrNoExtend(PatternRewriter &rewriter, Location loc, Value ptr,
                            Value offset, RegisterTypeInterface resultTy) {
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());
  Value addDst = createAllocation(rewriter, loc, resultTy);
  return lsir::AddIOp::create(rewriter, loc, i32Semantics, addDst, ptr, offset)
      .getDstRes();
}

LogicalResult
PtrAddOpPattern::matchAndRewrite(PtrAddOp op, PatternRewriter &rewriter) const {
  TypedValue<AMDGCNRegisterTypeInterface> ptr = op.getPtr();
  TypedValue<VGPRType> dynamicOffset = op.getDynamicOffset();
  TypedValue<SGPRType> uniformOffset = op.getUniformOffset();
  int64_t constOffset = op.getConstOffset();

  // Bail if the offsets are not 32-bit.
  if (uniformOffset && uniformOffset.getType().getAsRange().size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "uniform offset must have register size 1");
  }
  if (dynamicOffset && dynamicOffset.getType().getAsRange().size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "dynamic offset must have register size 1");
  }

  // Trivial case: no offsets.
  if (!dynamicOffset && !uniformOffset && constOffset == 0) {
    rewriter.replaceOp(op, ptr);
    return success();
  }

  // Get the types and context.
  AMDGCNRegisterTypeInterface ptrTy = ptr.getType();
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  RegisterTypeInterface resultTy = op.getResult().getType();
  bool ptrIsSGPR = ptrTy.getRegisterKind() == RegisterKind::SGPR;
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());

  // Compute uniform + const (i32) or null.
  Value uniformVal;
  if (uniformOffset && constOffset != 0) {
    Value cst =
        getI32Constant(rewriter, loc, static_cast<int32_t>(constOffset));
    Value dst = createAllocation(rewriter, loc, getSGPR(ctx, 1));
    uniformVal = lsir::AddIOp::create(rewriter, loc, i32Semantics, dst,
                                      uniformOffset, cst)
                     .getDstRes();
  } else if (uniformOffset) {
    uniformVal = uniformOffset;
  } else if (constOffset != 0) {
    uniformVal =
        getI32Constant(rewriter, loc, static_cast<int32_t>(constOffset));
  }

  // If ptr is SGPR, compute (ptr + signExtendToI64(uniform + const)) + dynamic.
  // TODO: Add flags to `ptr_add` as we then could avoid sign extension.
  if (ptrIsSGPR) {
    Value base = ptr;
    if (uniformVal) {
      // If the offset is a signless integer, we need to put it in an SGPR.
      if (uniformVal.getType().isSignlessInteger()) {
        Value alloc = createAllocation(rewriter, loc, getSGPR(ctx, 1));
        uniformVal =
            lsir::MovOp::create(rewriter, loc, alloc, uniformVal).getDstRes();
      }
      base = addPtrWithSignExtend(rewriter, loc, base, uniformVal, ptrTy,
                                  getSGPR(ctx, 2));
    }
    if (dynamicOffset) {
      base = addPtrWithSignExtend(rewriter, loc, base, dynamicOffset, resultTy,
                                  getVGPR(ctx, 2));
    }
    rewriter.replaceOp(op, base);
    return success();
  }

  // The ptr is a VGPR, compute (ptr + ((uniform + const) + dynamic)).
  Value offsetVal = uniformVal;

  if (offsetVal && dynamicOffset) {
    offsetVal =
        lsir::AddIOp::create(rewriter, loc, i32Semantics,
                             createAllocation(rewriter, loc, getVGPR(ctx, 1)),
                             offsetVal, dynamicOffset)
            .getDstRes();
  } else if (dynamicOffset) {
    offsetVal = dynamicOffset;
  }
  assert(offsetVal && "offsetVal must be non-null");

  // If the ptr is a single VGPR we do a simple addition without sign extension.
  if (ptrTy.getAsRange().size() == 1) {
    Value result = addPtrNoExtend(rewriter, loc, ptr, offsetVal, resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  // If the offset is a signless integer, we need to put it in an SGPR.
  if (offsetVal.getType().isSignlessInteger()) {
    Value alloc = createAllocation(rewriter, loc, getSGPR(ctx, 1));
    offsetVal =
        lsir::MovOp::create(rewriter, loc, alloc, offsetVal).getDstRes();
  }

  // Compute the result.
  Value result = addPtrWithSignExtend(rewriter, loc, ptr, offsetVal, resultTy,
                                      getVGPR(ctx, 2));
  rewriter.replaceOp(op, result);
  return success();
}

/// Emit a scalar select instruction; returns null Value for unsupported sizes.
static Value emitScalarSelect(PatternRewriter &rewriter, Location loc,
                              int16_t rangeSize, Value dst, Value trueVal,
                              Value falseVal, Value flagReg) {
  if (rangeSize == 1)
    return SCselectB32::create(rewriter, loc, dst, trueVal, falseVal, flagReg)
        .getDst0Res();
  if (rangeSize == 2)
    return SCselectB64::create(rewriter, loc, dst, trueVal, falseVal, flagReg)
        .getDst0Res();
  return Value();
}

/// Emit a vector select instruction; returns null Value for unsupported sizes.
static Value emitVectorSelect(PatternRewriter &rewriter, Location loc,
                              int16_t rangeSize, Value dst, Value trueVal,
                              Value falseVal, Value vccMask) {
  if (rangeSize == 1) {
    // VOP2 src1 must be a VGPR; materialize via v_mov_b32 into dst if needed.
    // The resulting dst WAR in v_cndmask_b32 is well-defined for VOP2.
    if (!isa<VGPRType>(trueVal.getType())) {
      VMovB32::create(rewriter, loc, dst, trueVal);
      trueVal = dst;
    }
    return VCndmaskB32::create(rewriter, loc, dst, falseVal, trueVal, vccMask)
        .getDst0Res();
  }
  if (rangeSize == 2) {
    RegisterTypeInterface oTy = cast<RegisterTypeInterface>(dst.getType());
    ValueRange dstR = splitRange(rewriter, loc, dst);
    ValueRange trueR = splitRange(rewriter, loc, trueVal);
    ValueRange falseR = splitRange(rewriter, loc, falseVal);
    Value dst0 = getElemOr(dstR, 0, dst);
    Value true0 = getElemOr(trueR, 0, trueVal);
    Value false0 = getElemOr(falseR, 0, falseVal);
    Value dst1 = getElemOr(dstR, 1, dst);
    Value true1 = getElemOr(trueR, 1, trueVal);
    Value false1 = getElemOr(falseR, 1, falseVal);
    // Materialize non-VGPR true halves before v_cndmask_b32.
    if (!isa<VGPRType>(true0.getType())) {
      VMovB32::create(rewriter, loc, dst0, true0);
      true0 = dst0;
    }
    if (!isa<VGPRType>(true1.getType())) {
      VMovB32::create(rewriter, loc, dst1, true1);
      true1 = dst1;
    }
    Value lo = VCndmaskB32::create(rewriter, loc, dst0, false0, true0, vccMask)
                   .getDst0Res();
    Value hi = VCndmaskB32::create(rewriter, loc, dst1, false1, true1, vccMask)
                   .getDst0Res();
    return MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi});
  }
  return Value();
}

/// Broadcast SCC (or other non-lane-mask flag) to a full lane mask.
static Value broadcastFlagToLaneMask(PatternRewriter &rewriter, Location loc,
                                     Operation *anchor, Value flagReg) {
  if (isWave32(anchor)) {
    Value vccDst = createAllocation(
        rewriter, loc, VCCLoType::get(rewriter.getContext(), Register()));
    Type i32 = rewriter.getI32Type();
    Value allOnes = arith::ConstantOp::create(rewriter, loc, i32,
                                              rewriter.getIntegerAttr(i32, -1));
    Value zero = arith::ConstantOp::create(rewriter, loc, i32,
                                           rewriter.getIntegerAttr(i32, 0));
    return SCselectB32::create(rewriter, loc, vccDst, allOnes, zero, flagReg)
        .getDst0Res();
  }
  Value vccDst = createAllocation(
      rewriter, loc, VCCType::get(rewriter.getContext(), Register()));
  Type i64 = rewriter.getI64Type();
  Value allOnes = arith::ConstantOp::create(rewriter, loc, i64,
                                            rewriter.getIntegerAttr(i64, -1));
  Value zero = arith::ConstantOp::create(rewriter, loc, i64,
                                         rewriter.getIntegerAttr(i64, 0));
  return SCselectB64::create(rewriter, loc, vccDst, allOnes, zero, flagReg)
      .getDst0Res();
}

//===----------------------------------------------------------------------===//
// SelectOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpPattern::matchAndRewrite(lsir::SelectOp op,
                                 PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value dst = op.getDst();
  Value flagReg = op.getCondition();
  Value trueVal = op.getTrueValue();
  Value falseVal = op.getFalseValue();

  bool resultIsVector = isVGPR(dst.getType(), /*numWords=*/-1) ||
                        isVGPR(trueVal.getType(), /*numWords=*/-1) ||
                        isVGPR(falseVal.getType(), /*numWords=*/-1);
  int16_t rangeSize =
      cast<RegisterTypeInterface>(dst.getType()).getAsRange().size();

  if (!resultIsVector) {
    // s_cselect_b32/b64: sdst = SCC ? src0 : src1.
    Value result = emitScalarSelect(rewriter, loc, rangeSize, dst, trueVal,
                                    falseVal, flagReg);
    if (!result)
      return rewriter.notifyMatchFailure(op, "unsupported scalar select width");
    rewriter.replaceOp(op, result);
    return success();
  }

  // v_cndmask_b32: vdst = VCC[lane] ? src1 : src0 (note: reversed order!).
  // src0 = false_value, src1 = true_value, src2 = lane mask.
  Value vccMask = isLaneMask(flagReg.getType())
                      ? flagReg
                      : broadcastFlagToLaneMask(rewriter, loc, op, flagReg);
  Value result = emitVectorSelect(rewriter, loc, rangeSize, dst, trueVal,
                                  falseVal, vccMask);
  if (!result)
    return rewriter.notifyMatchFailure(op, "unsupported vector select width");
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOpPattern helpers
//===----------------------------------------------------------------------===//

static Value createScalarCompare(PatternRewriter &rewriter, Location loc,
                                 arith::CmpIPredicate pred, Value dst,
                                 Value lhs, Value rhs) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return SCmpEqI32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::ne:
    return SCmpLgI32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::slt:
    return SCmpLtI32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::sle:
    return SCmpLeI32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::sgt:
    return SCmpGtI32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::sge:
    return SCmpGeI32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::ult:
    return SCmpLtU32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::ule:
    return SCmpLeU32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::ugt:
    return SCmpGtU32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  case arith::CmpIPredicate::uge:
    return SCmpGeU32::create(rewriter, loc, dst, lhs, rhs).getSccDstRes();
  }
  llvm_unreachable("unknown CmpIPredicate");
}

/// Create a v_cmp_* instruction for vector comparisons. The 32-bit VOPC
/// encoding requires rhs (src1) to be a VGPR; swap operands before calling if
/// needed.
static Value createVectorCompare(PatternRewriter &rewriter, Location loc,
                                 arith::CmpIPredicate pred, Value dst,
                                 Value lhs, Value rhs) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return VCmpEqI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::ne:
    return VCmpNeI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::slt:
    return VCmpLtI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::sle:
    return VCmpLeI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::sgt:
    return VCmpGtI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::sge:
    return VCmpGeI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::ult:
    return VCmpLtU32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::ule:
    return VCmpLeU32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::ugt:
    return VCmpGtU32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  case arith::CmpIPredicate::uge:
    return VCmpGeU32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  }
  llvm_unreachable("unknown CmpIPredicate");
}

/// Swap a comparison predicate (a op b becomes b swapped_op a).
static arith::CmpIPredicate swapPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unknown CmpIPredicate");
}

//===----------------------------------------------------------------------===//
// CmpIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult CmpIOpPattern::matchAndRewrite(lsir::CmpIOp op,
                                             PatternRewriter &rewriter) const {
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  arith::CmpIPredicate pred = op.getPredicate();
  Location loc = op.getLoc();

  Value result;
  if (isLaneMask(dst.getType())) {
    if (!isa<VGPRType>(rhs.getType())) {
      assert(isa<VGPRType>(lhs.getType()) &&
             "at least one operand must be a VGPR for vector compare");
      std::swap(lhs, rhs);
      pred = swapPredicate(pred);
    }
    result = createVectorCompare(rewriter, loc, pred, dst, lhs, rhs);
  } else {
    // Scalar compare: s_cmp_* writes to SCC.
    result = createScalarCompare(rewriter, loc, pred, dst, lhs, rhs);
  }

  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// DivUI/RemUI/DivSI/RemSI helpers
//===----------------------------------------------------------------------===//

/// Allocate a lane-mask (VCC or VCC_LO) depending on wave width.
static Value allocLaneMask(PatternRewriter &rewriter, Location loc,
                           Operation *op) {
  Type lmt = getLaneMaskType(op);
  return createAllocation(rewriter, loc, cast<RegisterTypeInterface>(lmt));
}

/// Ensure a Value is in a VGPR; materialise via v_mov_b32 if needed.
static Value toVGPR(PatternRewriter &rewriter, Location loc, Value v) {
  if (isa<VGPRType>(v.getType()))
    return v;
  Value dst = createAllocation(rewriter, loc, getVGPR(rewriter.getContext()));
  return VMovB32::create(rewriter, loc, dst, v).getDst0Res();
}

/// Emit v_cndmask_b32: result = mask ? trueVal : falseVal.
/// VOP2 requires src1 (trueVal) to be a VGPR; promotes via toVGPR if needed.
static Value emitCndmask(PatternRewriter &rewriter, Location loc, Value dst,
                         Value falseVal, Value trueVal, Value mask) {
  trueVal = toVGPR(rewriter, loc, trueVal);
  return VCndmaskB32::create(rewriter, loc, dst, falseVal, trueVal, mask)
      .getDst0Res();
}

/// Compute the integer reciprocal estimate Z ≈ 2^32/Y using one UNR step.
/// Matches the LLVM AMDGPULegalizerInfo::legalizeUnsignedDIV_REM32Impl logic.
static Value emitReciprocalEstimate(PatternRewriter &rewriter, Location loc,
                                    Value divisor) {
  MLIRContext *ctx = rewriter.getContext();
  std::function<Value()> vgpr = [&]() {
    return createAllocation(rewriter, loc, getVGPR(ctx));
  };

  // FloatY = v_cvt_f32_u32(Y).
  Value floatY =
      VCvtF32U32::create(rewriter, loc, vgpr(), divisor).getDst0Res();
  // Rcp = v_rcp_iflag_f32(FloatY).
  Value rcp = VRcpIflagF32::create(rewriter, loc, vgpr(), floatY).getDst0Res();
  // ScaledY = v_mul_f32(Scale, Rcp) where Scale = 0x4f7ffffe as f32.
  // Materialize the bit-pattern as an i32 literal via v_mov_b32 so no f32
  // arith.constant remains in the AMDGCN kernel (the translator only handles
  // integer constants, not float ones).
  Value scaleInt =
      getI32Constant(rewriter, loc, static_cast<int32_t>(0x4f7ffffe));
  Value scaleVgpr =
      VMovB32::create(rewriter, loc, vgpr(), scaleInt).getDst0Res();
  Value scaledY =
      VMulF32::create(rewriter, loc, vgpr(), scaleVgpr, rcp).getDst0Res();
  // Z = v_cvt_u32_f32(ScaledY).
  Value z = VCvtU32F32::create(rewriter, loc, vgpr(), scaledY).getDst0Res();

  // One UNR step: NegY = 0 - Y; NegYZ = NegY * Z; Z = Z + mulhi(Z, NegYZ).
  Value negY = VSubU32::create(rewriter, loc, vgpr(),
                               getI32Constant(rewriter, loc, 0), divisor)
                   .getDst0Res();
  Value negYZ = VMulLoU32::create(rewriter, loc, vgpr(), negY, z).getDst0Res();
  Value mulHiZNegYZ =
      VMulHiU32::create(rewriter, loc, vgpr(), z, negYZ).getDst0Res();
  z = VAddU32::create(rewriter, loc, vgpr(), z, mulHiZNegYZ).getDst0Res();
  return z;
}

/// One refinement round: if R >= Y then Q += 1, R -= Y.
/// Returns the updated {Q, R}.
static std::pair<Value, Value> emitRefinementStep(PatternRewriter &rewriter,
                                                  Location loc, Operation *op,
                                                  Value q, Value r,
                                                  Value divisor) {
  MLIRContext *ctx = rewriter.getContext();
  std::function<Value()> vgpr = [&]() {
    return createAllocation(rewriter, loc, getVGPR(ctx));
  };

  // Cond = R >= Y (unsigned).
  Value cond = VCmpGeU32::create(rewriter, loc,
                                 allocLaneMask(rewriter, loc, op), r, divisor)
                   .getDst0Res();
  // Q = Cond ? Q + 1 : Q.
  Value qP1 = VAddU32::create(rewriter, loc, vgpr(),
                              getI32Constant(rewriter, loc, 1), q)
                  .getDst0Res();
  q = emitCndmask(rewriter, loc, vgpr(), q, qP1, cond);
  // R = Cond ? R - Y : R.
  Value rMinusY =
      VSubU32::create(rewriter, loc, vgpr(), r, divisor).getDst0Res();
  r = emitCndmask(rewriter, loc, vgpr(), r, rMinusY, cond);
  return {q, r};
}

/// Full unsigned 32-bit divide/remainder matching LLVM's reference algorithm.
/// Returns {quotient, remainder}; dividend and divisor must be VGPR i32 values.
static std::pair<Value, Value> emitUnsignedDivRem(PatternRewriter &rewriter,
                                                  Location loc, Operation *op,
                                                  Value dividend,
                                                  Value divisor) {
  MLIRContext *ctx = rewriter.getContext();
  std::function<Value()> vgpr = [&]() {
    return createAllocation(rewriter, loc, getVGPR(ctx));
  };

  // Z ≈ 2^32 / divisor via reciprocal estimate and one UNR step.
  Value z = emitReciprocalEstimate(rewriter, loc, divisor);
  // Q = mulhi(X, Z) — initial quotient estimate.
  Value q = VMulHiU32::create(rewriter, loc, vgpr(), dividend, z).getDst0Res();
  // R = X - Q * Y — initial remainder estimate.
  Value qy = VMulLoU32::create(rewriter, loc, vgpr(), q, divisor).getDst0Res();
  Value r = VSubU32::create(rewriter, loc, vgpr(), dividend, qy).getDst0Res();
  // Two refinement rounds to correct off-by-one errors.
  std::tie(q, r) = emitRefinementStep(rewriter, loc, op, q, r, divisor);
  std::tie(q, r) = emitRefinementStep(rewriter, loc, op, q, r, divisor);
  return {q, r};
}

/// Signed 32-bit divide/remainder via sign-correction around the unsigned core.
static std::pair<Value, Value> emitSignedDivRem(PatternRewriter &rewriter,
                                                Location loc, Operation *op,
                                                Value dividend, Value divisor) {
  MLIRContext *ctx = rewriter.getContext();
  std::function<Value()> vgpr = [&]() {
    return createAllocation(rewriter, loc, getVGPR(ctx));
  };

  Value c31 = getI32Constant(rewriter, loc, 31);
  // sa = arithmetic right-shift of dividend by 31: all-ones if negative.
  // VAshrrevI32 operands are reversed: src0=shift-amount, src1=value.
  Value sa =
      VAshrrevI32::create(rewriter, loc, vgpr(), c31, dividend).getDst0Res();
  // sb = arithmetic right-shift of divisor by 31.
  Value sb =
      VAshrrevI32::create(rewriter, loc, vgpr(), c31, divisor).getDst0Res();
  // absA = (dividend ^ sa) - sa — two's-complement absolute value.
  Value xorA =
      VXorB32::create(rewriter, loc, vgpr(), sa, dividend).getDst0Res();
  Value absA = VSubU32::create(rewriter, loc, vgpr(), xorA, sa).getDst0Res();
  // absB = (divisor ^ sb) - sb.
  Value xorB = VXorB32::create(rewriter, loc, vgpr(), sb, divisor).getDst0Res();
  Value absB = VSubU32::create(rewriter, loc, vgpr(), xorB, sb).getDst0Res();
  auto [uq, ur] = emitUnsignedDivRem(rewriter, loc, op, absA, absB);
  // qsign = sa ^ sb — quotient sign mask.
  Value qsign = VXorB32::create(rewriter, loc, vgpr(), sa, sb).getDst0Res();
  // q = (uq ^ qsign) - qsign — re-apply sign.
  Value qxor = VXorB32::create(rewriter, loc, vgpr(), qsign, uq).getDst0Res();
  Value q = VSubU32::create(rewriter, loc, vgpr(), qxor, qsign).getDst0Res();
  // r = (ur ^ sa) - sa — remainder inherits dividend's sign.
  Value rxor = VXorB32::create(rewriter, loc, vgpr(), sa, ur).getDst0Res();
  Value r = VSubU32::create(rewriter, loc, vgpr(), rxor, sa).getDst0Res();
  return {q, r};
}

//===----------------------------------------------------------------------===//
// DivUIOpPattern / RemUIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult DivUIOpPattern::matchAndRewrite(lsir::DivUIOp op,
                                              PatternRewriter &rewriter) const {
  if (op.getSemantics().getWidth() != 32)
    return rewriter.notifyMatchFailure(
        op, "only 32-bit integer divide is supported");
  Value dst = op.getDst();
  if (!isa<VGPRType>(dst.getType()))
    return rewriter.notifyMatchFailure(
        op, "nyi: non-VGPR destination for integer divide");
  Location loc = op.getLoc();
  Value dividend = toVGPR(rewriter, loc, op.getLhs());
  Value divisor = toVGPR(rewriter, loc, op.getRhs());
  auto [q, r] =
      emitUnsignedDivRem(rewriter, loc, op.getOperation(), dividend, divisor);
  Value result = lsir::CopyOp::create(rewriter, loc, dst, q).getTargetRes();
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult RemUIOpPattern::matchAndRewrite(lsir::RemUIOp op,
                                              PatternRewriter &rewriter) const {
  if (op.getSemantics().getWidth() != 32)
    return rewriter.notifyMatchFailure(
        op, "only 32-bit integer remainder is supported");
  Value dst = op.getDst();
  if (!isa<VGPRType>(dst.getType()))
    return rewriter.notifyMatchFailure(
        op, "nyi: non-VGPR destination for integer remainder");
  Location loc = op.getLoc();
  Value dividend = toVGPR(rewriter, loc, op.getLhs());
  Value divisor = toVGPR(rewriter, loc, op.getRhs());
  auto [q, r] =
      emitUnsignedDivRem(rewriter, loc, op.getOperation(), dividend, divisor);
  Value result = lsir::CopyOp::create(rewriter, loc, dst, r).getTargetRes();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// DivSIOpPattern / RemSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult DivSIOpPattern::matchAndRewrite(lsir::DivSIOp op,
                                              PatternRewriter &rewriter) const {
  if (op.getSemantics().getWidth() != 32)
    return rewriter.notifyMatchFailure(
        op, "only 32-bit signed integer divide is supported");
  Value dst = op.getDst();
  if (!isa<VGPRType>(dst.getType()))
    return rewriter.notifyMatchFailure(
        op, "nyi: non-VGPR destination for signed integer divide");
  Location loc = op.getLoc();
  Value dividend = toVGPR(rewriter, loc, op.getLhs());
  Value divisor = toVGPR(rewriter, loc, op.getRhs());
  auto [q, r] =
      emitSignedDivRem(rewriter, loc, op.getOperation(), dividend, divisor);
  Value result = lsir::CopyOp::create(rewriter, loc, dst, q).getTargetRes();
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult RemSIOpPattern::matchAndRewrite(lsir::RemSIOp op,
                                              PatternRewriter &rewriter) const {
  if (op.getSemantics().getWidth() != 32)
    return rewriter.notifyMatchFailure(
        op, "only 32-bit signed integer remainder is supported");
  Value dst = op.getDst();
  if (!isa<VGPRType>(dst.getType()))
    return rewriter.notifyMatchFailure(
        op, "nyi: non-VGPR destination for signed integer remainder");
  Location loc = op.getLoc();
  Value dividend = toVGPR(rewriter, loc, op.getLhs());
  Value divisor = toVGPR(rewriter, loc, op.getRhs());
  auto [q, r] =
      emitSignedDivRem(rewriter, loc, op.getOperation(), dividend, divisor);
  Value result = lsir::CopyOp::create(rewriter, loc, dst, r).getTargetRes();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ToAMDGCNPass patterns
//===----------------------------------------------------------------------===//

void mlir::aster::amdgcn::populateToAMDGCNPatterns(
    RewritePatternSet &patterns) {
  patterns.add< // Arithmetic ops.
      AddFOpPattern, AddIOpPattern, AndIOpPattern, CmpIOpPattern,
      SelectOpPattern, ExtFOpPattern, ExtSIOpPattern, ExtUIOpPattern,
      TruncFOpPattern, SIToFPOpPattern, UIToFPOpPattern, FPToSIOpPattern,
      FPToUIOpPattern, MaximumFOpPattern, MinimumFOpPattern, MulFOpPattern,
      MulIOpPattern, MulHiSIOpPattern, OrIOpPattern, ShLIOpPattern,
      ShRSIOpPattern, ShRUIOpPattern, SubFOpPattern, SubIOpPattern,
      TruncIOpPattern, XOrIOpPattern, DivUIOpPattern, RemUIOpPattern,
      DivSIOpPattern, RemSIOpPattern,
      // Memory ops.
      AllocaOpPattern, AssumeNoaliasOpPattern, LoadOpPattern, StoreOpPattern,
      // Data movement ops.
      MovOpPattern, RegCastOpPattern,
      // Pointer ops.
      PtrAddOpPattern,
      // Control ops.
      KernelOpPattern, ReturnOpPattern>(patterns.getContext());
}
