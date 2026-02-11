//===- RegisterInterferenceGraph.h - Register interference graph -*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCEGRAPH_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCEGRAPH_H

#include "aster/Support/Graph.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class Operation;
class DataFlowSolver;
class SymbolTableCollection;
} // namespace mlir

namespace mlir::aster::amdgcn {
/// Register interference graph for AMDGCN register allocation. This analysis
/// uses the RegisterLiveness analysis to build an interference graph where
/// nodes represent allocations and edges connect allocations that overlap in
/// time.
struct RegisterInterferenceGraph : public Graph {
  /// Create the interference graph for the given operation.
  /// This will load and run RegisterLiveness internally.
  static FailureOr<RegisterInterferenceGraph>
  create(Operation *op, DataFlowSolver &solver,
         SymbolTableCollection &symbolTable);

  /// Print the interference graph.
  void print(raw_ostream &os) const;

  /// Get the node ID for a value. Returns -1 if not found.
  NodeID getNodeId(Value value) const;

  /// Get the value for a node ID.
  Value getValue(NodeID nodeId) const;

  /// Get all values in the graph.
  ArrayRef<Value> getValues() const { return values; }
  MutableArrayRef<Value> getValues() { return values; }

  /// Get the canonical (root) node ID for a possibly-merged node.
  /// Returns the input ID if the node was never merged.
  NodeID getCanonical(NodeID id) const;

  /// Get all node IDs that were merged into the given canonical node.
  SmallVector<NodeID> getMergedNodes(NodeID canonicalId) const;

private:
  RegisterInterferenceGraph() : Graph(/*directed=*/false) {}

  /// Run the interference analysis on the given operation.
  LogicalResult run(Operation *op, DataFlowSolver &solver);

  /// Handle a generic operation during graph construction.
  LogicalResult handleOp(Operation *op, DataFlowSolver &solver);

  /// Add edges between allocations.
  void addEdges(Value lhs, Value rhs);

  /// Add edges between all the related pairs of allocations in the given list.
  void addEdges(SmallVectorImpl<Value> &allocas);

  /// Get or create a node ID for an allocation.
  NodeID getOrCreateNodeId(Value allocation);

  /// Merge node `remove` into node `keep`. All future lookups for values
  /// that mapped to `remove` will return `keep`.
  void mergeNodes(NodeID keep, NodeID remove);

  /// Get the canonical (root) node ID, following merge chains.
  NodeID canonical(NodeID id) const;

  SmallVector<Value> values;
  llvm::DenseMap<Value, NodeID> valueToNodeId;
  /// Merge parent map: merged[id] = parent. Root nodes point to themselves.
  SmallVector<NodeID> mergeParent;
};

} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_REGISTERINTERFERENCEGRAPH_H
