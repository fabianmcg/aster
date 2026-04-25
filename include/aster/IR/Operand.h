//===- Operand.h - Aster operand wrapper -------------------------*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Operand wrapper type.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_OPERAND_H
#define ASTER_IR_OPERAND_H

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

namespace mlir::aster {
/// A nullable wrapper around an OpOperand pointer.
struct Operand {
  Operand() = default;
  Operand(OpOperand &operand) : operand(&operand) {}

  explicit operator bool() const { return operand != nullptr; }
  operator Value() const { return operand ? operand->get() : Value(); }
  operator OpOperand *() const { return operand; }
  OpOperand *operator->() const { return operand; }

  /// Get the underlying OpOperand pointer.
  OpOperand *get() const { return operand; }
  /// Get the value of the operand, or a null Value if the operand is not set.
  Value value() const { return operand ? operand->get() : Value(); }
  /// Get the type of the operand, or a null Type if the operand is not set.
  Type type() const {
    if (Value val = value())
      return val.getType();
    return Type();
  }

private:
  OpOperand *operand = nullptr;
};

} // namespace mlir::aster

#endif // ASTER_IR_OPERAND_H
