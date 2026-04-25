//===- StoreOpInterface.td - Store Op Interface --------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the store operation interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_STOREOPINTERFACE_TD
#define ASTER_INTERFACES_STOREOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def StoreOpInterface : OpInterface<"StoreOpInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for store operations providing uniform access to operands and
    results.
  }];
  let methods = [
    InterfaceMethod<"Get the data operand.",
      "::mlir::aster::Operand", "getData", (ins), [{
        return $_op.getDataMutable();
      }]>,
    InterfaceMethod<"Get the address operand.",
      "::mlir::aster::Operand", "getAddr", (ins), [{
        return $_op.getAddrMutable();
      }]>,
    InterfaceMethod<"Get the uniform offset operand.",
      "::mlir::aster::Operand", "getUniformOffset", (ins), [{
        ::mlir::MutableOperandRange r = $_op.getUniformOffsetMutable();
        return r.size() ? ::mlir::aster::Operand(r[0]) : ::mlir::aster::Operand();
      }]>,
    InterfaceMethod<"Get the dynamic offset operand.",
      "::mlir::aster::Operand", "getDynamicOffset", (ins), [{
        ::mlir::MutableOperandRange r = $_op.getDynamicOffsetMutable();
        return r.size() ? ::mlir::aster::Operand(r[0]) : ::mlir::aster::Operand();
      }]>,
    InterfaceMethod<"Get the constant offset operand.",
      "::mlir::aster::Operand", "getConstantOffset", (ins), [{
        ::mlir::MutableOperandRange r = $_op.getConstantOffsetMutable();
        return r.size() ? ::mlir::aster::Operand(r[0]) : ::mlir::aster::Operand();
      }]>,
    InterfaceMethod<"Get the token result.",
      "::mlir::OpResult", "getToken", (ins), [{
        return ::mlir::cast<::mlir::OpResult>(::mlir::Value($_op.getToken()));
      }]>
  ];
}

#endif // ASTER_INTERFACES_STOREOPINTERFACE_TD
