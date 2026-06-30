//===- CodeGenPatterns.cpp ------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGCN CodeGen patterns
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Ptr/IR/PtrEnums.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct IDDimOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrLoadOpPattern
//===----------------------------------------------------------------------===//
struct PtrLoadOpPattern : public OpCodeGenPattern<ptr::LoadOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(ptr::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrStoreOpPattern
//===----------------------------------------------------------------------===//
struct PtrStoreOpPattern : public OpCodeGenPattern<ptr::StoreOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(ptr::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrAddOpPattern (ptr::PtrAddOp -> amdgcn::PtrAddOp)
//===----------------------------------------------------------------------===//
struct PtrAddOpPattern : public OpCodeGenPattern<ptr::PtrAddOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(ptr::PtrAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// GetLDSOffsetOpPattern (i32/index result -> sgpr result)
//===----------------------------------------------------------------------===//
struct GetLDSOffsetOpPattern : public OpCodeGenPattern<amdgcn::GetLDSOffsetOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(amdgcn::GetLDSOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// GetCfMaskCodeGenPattern
//===----------------------------------------------------------------------===//
struct GetCfMaskCodeGenPattern
    : public OpCodeGenPattern<aster_utils::GetCfMaskOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(aster_utils::GetCfMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SetCfMaskCodeGenPattern
//===----------------------------------------------------------------------===//
struct SetCfMaskCodeGenPattern
    : public OpCodeGenPattern<aster_utils::SetCfMaskOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(aster_utils::SetCfMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult IDDimOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Type regTy = std::is_same_v<OpTy, aster_utils::ThreadIdOp>
                   ? Type(amdgcn::VGPRType::get(op.getContext(), Register()))
                   : Type(amdgcn::SGPRType::get(op.getContext(), Register()));
  auto nOp = NewOpTy::create(
      rewriter, op.getLoc(), regTy,
      static_cast<amdgcn::Dim>(static_cast<int8_t>(op.getDim())));
  rewriter.replaceOpWithNewOp<lsir::RegCastOp>(op, type, nOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

/// Returns true if the ptr type has local (LDS) address space.
static bool isLocalMemory(ptr::PtrType ptrType) {
  auto memSpace = ptrType.getMemorySpace();
  if (!memSpace)
    return false;
  auto addrSpace = dyn_cast<AddressSpaceAttr>(memSpace);
  return addrSpace && addrSpace.getSpace() == AddressSpaceKind::Local;
}

/// Create an i32 constant value.
static Value getI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getIntegerAttr(builder.getI32Type(), value));
}

/// Create a DS_READ instruction for the given number of 32-bit words.
static FailureOr<Value> createDSRead(OpBuilder &rewriter, Location loc,
                                     Value dst, Value addr, int64_t numWords) {
  Value offset = getI32Constant(rewriter, loc, 0);
  // Trailing optionals (gds, fence_token) must be passed explicitly: a codegen
  // ds_read has no GDS flag and takes no fence token.
  UnitAttr noGds = {};
  Value noFenceToken = {};
  switch (numWords) {
  case 1:
    return DsReadB32::create(rewriter, loc, dst, addr, offset, noGds,
                             noFenceToken)
        .getDestRes();
  case 2:
    return DsReadB64::create(rewriter, loc, dst, addr, offset, noGds,
                             noFenceToken)
        .getDestRes();
  case 3:
    return DsReadB96::create(rewriter, loc, dst, addr, offset, noGds,
                             noFenceToken)
        .getDestRes();
  case 4:
    return DsReadB128::create(rewriter, loc, dst, addr, offset, noGds,
                              noFenceToken)
        .getDestRes();
  default:
    return failure();
  }
}

/// Create a GLOBAL_LOAD instruction for the given number of 32-bit words.
static FailureOr<Value> createGlobalLoad(OpBuilder &rewriter, Location loc,
                                         Value dst, Value addr,
                                         int64_t numWords,
                                         int64_t numBytes = -1) {
  Value cOff = getI32Constant(rewriter, loc, 0);
  // Use a 16-bit load when the element is exactly 2 bytes.
  if (numWords == 1 && numBytes == 2)
    return GlobalLoadUshort::create(rewriter, loc, dst, addr, nullptr, cOff)
        .getDestRes();
  switch (numWords) {
  case 1:
    return GlobalLoadDword::create(rewriter, loc, dst, addr, nullptr, cOff)
        .getDestRes();
  case 2:
    return GlobalLoadDwordx2::create(rewriter, loc, dst, addr, nullptr, cOff)
        .getDestRes();
  case 3:
    return GlobalLoadDwordx3::create(rewriter, loc, dst, addr, nullptr, cOff)
        .getDestRes();
  case 4:
    return GlobalLoadDwordx4::create(rewriter, loc, dst, addr, nullptr, cOff)
        .getDestRes();
  default:
    return failure();
  }
}

/// Create a DS_WRITE instruction for the given number of 32-bit words.
static LogicalResult createDSWrite(OpBuilder &rewriter, Location loc,
                                   Value data, Value addr, int64_t numWords) {
  Value offset = getI32Constant(rewriter, loc, 0);
  switch (numWords) {
  case 1:
    DsWriteB32::create(rewriter, loc, addr, data, offset);
    return success();
  case 2:
    DsWriteB64::create(rewriter, loc, addr, data, offset);
    return success();
  case 3:
    DsWriteB96::create(rewriter, loc, addr, data, offset);
    return success();
  case 4:
    DsWriteB128::create(rewriter, loc, addr, data, offset);
    return success();
  default:
    return failure();
  }
}

/// Create a GLOBAL_STORE instruction for the given number of 32-bit words.
static LogicalResult createGlobalStore(OpBuilder &rewriter, Location loc,
                                       Value data, Value addr, int64_t numWords,
                                       int64_t numBytes = -1) {
  Value cOff = getI32Constant(rewriter, loc, 0);
  Value dOff = nullptr;
  if (isSGPR(addr.getType(), 0))
    dOff = lsir::MovOp::create(
               rewriter, loc,
               createAllocation(rewriter, loc, getVGPR(rewriter.getContext())),
               cOff)
               .getDstRes();
  // Use a 16-bit store when the element is exactly 2 bytes.
  if (numWords == 1 && numBytes == 2) {
    GlobalStoreShort::create(rewriter, loc, data, addr, dOff, cOff);
    return success();
  }
  switch (numWords) {
  case 1:
    GlobalStoreDword::create(rewriter, loc, data, addr, dOff, cOff);
    return success();
  case 2:
    GlobalStoreDwordx2::create(rewriter, loc, data, addr, dOff, cOff);
    return success();
  case 3:
    GlobalStoreDwordx3::create(rewriter, loc, data, addr, dOff, cOff);
    return success();
  case 4:
    GlobalStoreDwordx4::create(rewriter, loc, data, addr, dOff, cOff);
    return success();
  default:
    return failure();
  }
}

//===----------------------------------------------------------------------===//
// PtrLoadOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
PtrLoadOpPattern::matchAndRewrite(ptr::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Type resultType = converter.convertType(op.getResult());
  int64_t numBytes = converter.getTypeSize(op.getResult().getType());
  int64_t numWords = (numBytes + 3) / 4;
  Value dst = createAlloca(rewriter, loc, resultType);
  Value addr = adaptor.getPtr();
  auto ptrType = cast<ptr::PtrType>(op.getPtr().getType());

  FailureOr<Value> result =
      isLocalMemory(ptrType)
          ? createDSRead(rewriter, loc, dst, addr, numWords)
          : createGlobalLoad(rewriter, loc, dst, addr, numWords, numBytes);
  if (failed(result))
    return rewriter.notifyMatchFailure(op,
                                       "unsupported word count for ptr.load");

  rewriter.replaceOp(op, *result);
  return success();
}

//===----------------------------------------------------------------------===//
// PtrStoreOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
PtrStoreOpPattern::matchAndRewrite(ptr::StoreOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value data = adaptor.getValue();
  int64_t numBytes = converter.getTypeSize(op.getValue().getType());
  int64_t numWords = (numBytes + 3) / 4;
  Value addr = adaptor.getPtr();
  auto ptrType = cast<ptr::PtrType>(op.getPtr().getType());

  if (data.getType().isIntOrIndexOrFloat()) {
    data =
        lsir::MovOp::create(rewriter, op.getLoc(),
                            createAllocation(rewriter, op.getLoc(),
                                             getVGPR(rewriter.getContext(), 1)),
                            data)
            .getDstRes();
  } else if (auto sgprTy = dyn_cast<SGPRType>(data.getType())) {
    data = lsir::CopyOp::create(
               rewriter, op.getLoc(),
               createAllocation(
                   rewriter, op.getLoc(),
                   getVGPR(rewriter.getContext(), sgprTy.getRange().size())),
               data)
               .getTargetRes();
  }

  LogicalResult created =
      isLocalMemory(ptrType)
          ? createDSWrite(rewriter, loc, data, addr, numWords)
          : createGlobalStore(rewriter, loc, data, addr, numWords, numBytes);
  if (failed(created))
    return rewriter.notifyMatchFailure(op,
                                       "unsupported word count for ptr.store");

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Internal functions
//===----------------------------------------------------------------------===//

/// Untangle unrealized conversion casts to find the original value.
static Value untangleConvertValue(Value value) {
  if (isa<RegisterTypeInterface>(value.getType()))
    return value;
  auto cOp =
      dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  while (cOp && cOp.getNumOperands() == 1) {
    Value value = cOp.getOperand(0);
    if (isa<RegisterTypeInterface>(value.getType()))
      return value;
    cOp =
        dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  }
  return value;
}

static Type convertAttrConstraintToType(Attribute constraint,
                                        int64_t numWords) {
  auto kind = dyn_cast<amdgcn::RegisterKindAttr>(constraint);
  if (!kind)
    return nullptr;
  switch (kind.getValue()) {
  case amdgcn::RegisterKind::SGPR:
    if (numWords == 1)
      return amdgcn::SGPRType::get(kind.getContext(), Register());
    return amdgcn::SGPRType::get(kind.getContext(),
                                 RegisterRange(Register(), numWords));
  case amdgcn::RegisterKind::VGPR:
    if (numWords == 1)
      return amdgcn::VGPRType::get(kind.getContext(), Register());
    return amdgcn::VGPRType::get(kind.getContext(),
                                 RegisterRange(Register(), numWords));
  case amdgcn::RegisterKind::AGPR:
    if (numWords == 1)
      return amdgcn::AGPRType::get(kind.getContext(), Register());
    return amdgcn::AGPRType::get(kind.getContext(),
                                 RegisterRange(Register(), numWords));
  default:
    assert(false && "nyi register kind");
  }
  return nullptr;
}

static Type convertTypeImpl(Value value, const CodeGenConverter &converter) {
  if (Operation *defOp = value.getDefiningOp();
      defOp && m_Constant().match(value.getDefiningOp()))
    return value.getType();
  value = untangleConvertValue(value);
  if (isa<RegisterTypeInterface>(value.getType()))
    return value.getType();

  // i1 values map to SCC (thread-uniform) or VCC (divergent).
  if (value.getType().isInteger(1)) {
    std::optional<bool> isUniform = converter.isThreadUniform(value);
    if (isUniform.has_value() && *isUniform)
      return amdgcn::SCCType::get(value.getContext(), Register());
    return amdgcn::getLaneMaskType(value);
  }

  int64_t typeSize = converter.getTypeSize(value.getType());
  int64_t numWords = (typeSize + 3) / 4;

  // If there is a register constraint, use it to determine the type.
  if (Attribute constraint =
          converter.getState().getRegisterConstraint(value)) {
    if (Type t = convertAttrConstraintToType(constraint, numWords))
      return t;
  }

  std::optional<bool> isUniform = converter.isThreadUniform(value);
  assert(isUniform.has_value() &&
         "Type conversion for value without known thread-uniformity");

  if (isUniform.has_value() && *isUniform) {
    if (numWords > 1)
      return amdgcn::SGPRType::get(value.getContext(),
                                   RegisterRange(Register(), numWords));
    return amdgcn::SGPRType::get(value.getContext(), Register());
  }
  if (numWords > 1)
    return amdgcn::VGPRType::get(value.getContext(),
                                 RegisterRange(Register(), numWords));
  return amdgcn::VGPRType::get(value.getContext(), Register());
}

static Type convertTypeImpl(Type type, const CodeGenConverter &converter) {
  if (isa<RegisterTypeInterface>(type))
    return type;
  int64_t typeSize = converter.getTypeSize(type);
  int64_t numWords = (typeSize + 3) / 4;
  if (numWords > 1)
    return amdgcn::VGPRType::get(type.getContext(),
                                 RegisterRange(Register(), numWords));
  return amdgcn::VGPRType::get(type.getContext(), Register());
}

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
PtrAddOpPattern::matchAndRewrite(ptr::PtrAddOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value ptr = adaptor.getBase();
  Value offset = adaptor.getOffset();
  Type resultType = converter.convertType(op.getResult());

  Value dynamicOffset;
  Value uniformOffset;
  int64_t constOffset = 0;

  llvm::APInt constAttr;
  if (Operation *defOp = offset.getDefiningOp();
      defOp && m_ConstantInt(&constAttr).match(defOp)) {
    constOffset = constAttr.getSExtValue();
  } else {
    if (isVGPR(offset.getType(), 0)) {
      dynamicOffset = offset;
    } else if (isSGPR(offset.getType(), 0)) {
      uniformOffset = offset;
      // Result type must match ptr when using uniform_offset (amdgcn.ptr_add
      // infers from ptr operand).
      resultType = ptr.getType();
    } else {
      return rewriter.notifyMatchFailure(op, "invalid offset");
    }
  }

  auto flags = op.getFlags();
  auto newOp = rewriter.replaceOpWithNewOp<amdgcn::PtrAddOp>(
      op, resultType, ptr, dynamicOffset, uniformOffset, constOffset);
  newOp.setFlags(flags);
  return success();
}

//===----------------------------------------------------------------------===//
// GetCfMaskCodeGenPattern
//===----------------------------------------------------------------------===//

LogicalResult GetCfMaskCodeGenPattern::matchAndRewrite(
    aster_utils::GetCfMaskOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  // wave32 -> 1 SGPR (i32); wave64 -> 2 SGPRs (i64).
  int16_t words = isWave32(op) ? 1 : 2;
  // EXEC is a composite register; createAllocation handles its split/join.
  RegisterTypeInterface execType =
      isWave32(op) ? RegisterTypeInterface(EXECLoType::get(ctx, Register(0)))
                   : RegisterTypeInterface(EXECType::get(ctx, Register(0)));
  Value exec = createAllocation(rewriter, loc, execType);
  Value sgprDst = createAlloca(rewriter, loc, getSGPR(ctx, words));
  Value sgprMask =
      lsir::CopyOp::create(rewriter, loc, sgprDst, exec).getTargetRes();
  // lsir.from_reg is consumed by FromToRegOpPattern in the same codegen pass.
  rewriter.replaceOpWithNewOp<lsir::FromRegOp>(op, op.getMask().getType(),
                                               sgprMask);
  return success();
}

//===----------------------------------------------------------------------===//
// SetCfMaskCodeGenPattern
//===----------------------------------------------------------------------===//

LogicalResult SetCfMaskCodeGenPattern::matchAndRewrite(
    aster_utils::SetCfMaskOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  bool wave32 = isWave32(op);
  int16_t words = wave32 ? 1 : 2;
  // EXEC is a composite register; createAllocation handles its split/join.
  RegisterTypeInterface execType =
      wave32 ? RegisterTypeInterface(EXECLoType::get(ctx, Register(0)))
             : RegisterTypeInterface(EXECType::get(ctx, Register(0)));

  Value condVal = adaptor.getCondition();
  if (!condVal) {
    // Case 1: no condition — EXEC = saved.
    // lsir.to_reg is consumed by FromToRegOpPattern in the same codegen pass.
    Value sgprMask = lsir::ToRegOp::create(rewriter, loc, getSGPR(ctx, words),
                                           adaptor.getMask())
                         .getResult();
    Value exec = createAllocation(rewriter, loc, execType);
    lsir::CopyOp::create(rewriter, loc, exec, sgprMask);
    rewriter.eraseOp(op);
    return success();
  }

  // Cases 2 and 3: EXEC = saved & cond  or  EXEC = saved & ~cond.
  // condVal is already converted to the lane-mask type (VCC / VCC_LO) by the
  // type converter for divergent i1 values.
  Value sgprMask = lsir::ToRegOp::create(rewriter, loc, getSGPR(ctx, words),
                                         adaptor.getMask())
                       .getResult();
  Value execDst = createAllocation(rewriter, loc, execType);
  Value sccDst = AllocaOp::create(rewriter, loc, SCCType::get(ctx, Register()));
  if (wave32) {
    if (op.getComplement())
      SAndn2B32::create(rewriter, loc, execDst, sccDst, sgprMask, condVal);
    else
      SAndB32::create(rewriter, loc, execDst, sccDst, sgprMask, condVal);
  } else {
    if (op.getComplement())
      SAndn2B64::create(rewriter, loc, execDst, sccDst, sgprMask, condVal);
    else
      SAndB64::create(rewriter, loc, execDst, sccDst, sgprMask, condVal);
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::aster::amdgcn::getDependentCodeGenDialects(
    DialectRegistry &registry) {
  registry.insert<amdgcn::AMDGCNDialect, lsir::LSIRDialect>();
}

// Legalize get_lds_offset to register now in place to a register-typed result
// instead of leaving for the later generic materialization to bridge with an
// unrealized_conversion_cast.
LogicalResult GetLDSOffsetOpPattern::matchAndRewrite(
    amdgcn::GetLDSOffsetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  assert(converter.isThreadUniform(op.getResult()) &&
         "LDS offset must be workgroup-uniform");
  Type regTy = Type(amdgcn::SGPRType::get(ctx, Register()));
  rewriter.replaceOpWithNewOp<amdgcn::GetLDSOffsetOp>(op, regTy,
                                                      adaptor.getBuffer());
  return success();
}

void mlir::aster::amdgcn::populateCodeGenPatterns(CodeGenConverter &converter,
                                                  RewritePatternSet &patterns,
                                                  ConversionTarget &target) {
  // Configure the conversion target.
  target.addLegalDialect<amdgcn::AMDGCNDialect>();

  // Add the type conversions.
  converter.addConversion(
      [&converter](Type type) { return convertTypeImpl(type, converter); });
  converter.addConversion(
      [&converter](Value value) { return convertTypeImpl(value, converter); });
  converter.addConversion([](TokenDependencyTypeInterface tok) { return tok; });
  // The LDS buffer handle passes through unchanged, LDS allocation will map to
  // actual buffers and offsets.
  converter.addConversion([](amdgcn::LDSBufferType buf) { return buf; });
  // Legalize get_lds_offset to register now in place to a register-typed result
  // instead of leaving for the later generic materialization to bridge with an
  // unrealized_conversion_cast.
  target.addDynamicallyLegalOp<amdgcn::GetLDSOffsetOp>(
      [](amdgcn::GetLDSOffsetOp op) {
        return isa<RegisterTypeInterface>(op.getResult().getType());
      });

  target.addIllegalOp<aster_utils::ThreadIdOp, aster_utils::BlockIdOp,
                      aster_utils::BlockDimOp, aster_utils::GridDimOp,
                      aster_utils::AssumeRangeOp, aster_utils::AssumeUniformOp,
                      aster_utils::GetCfMaskOp, aster_utils::SetCfMaskOp,
                      lsir::FromRegOp, lsir::ToRegOp, lsir::RegConstraintOp,
                      ptr::LoadOp, ptr::StoreOp, ptr::PtrAddOp>();

  // Add the patterns.
  patterns.add<IDDimOpPattern<aster_utils::ThreadIdOp, amdgcn::ThreadIdOp>,
               IDDimOpPattern<aster_utils::BlockIdOp, amdgcn::BlockIdOp>,
               IDDimOpPattern<aster_utils::BlockDimOp, amdgcn::BlockDimOp>,
               IDDimOpPattern<aster_utils::GridDimOp, amdgcn::GridDimOp>,
               PtrLoadOpPattern, PtrStoreOpPattern, PtrAddOpPattern,
               GetLDSOffsetOpPattern, GetCfMaskCodeGenPattern,
               SetCfMaskCodeGenPattern>(converter);
}
