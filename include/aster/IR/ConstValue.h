//===- ConstValue.h - ConstValue --------------------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_CONSTVALUE_H
#define ASTER_IR_CONSTVALUE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/PointerUnion.h"

namespace mlir::aster {
/// A compile-time constant operand: either an MLIR Type (from a live value) or
/// a TypedAttr (from a constant value). A null PointerUnion represents an
/// absent optional operand.
using ConstValue = llvm::PointerUnion<mlir::Type, mlir::TypedAttr>;

/// Extract the MLIR Type from a ConstValue. Returns a null Type for a null
/// PointerUnion (absent optional operand).
inline mlir::Type getTypeOrValue(ConstValue cv) {
  if (cv.is<mlir::Type>())
    return cv.get<mlir::Type>();
  if (cv.is<mlir::TypedAttr>())
    return cv.get<mlir::TypedAttr>().getType();
  return mlir::Type{};
}
} // namespace mlir::aster

#endif // ASTER_IR_CONSTVALUE_H
