add_mlir_dialect_library(AsterUtilsDialect
  AsterUtilsAttrs.cpp
  AsterUtilsOps.cpp
  AsterUtilsTypes.cpp
  IntRangeImpl.cpp

  DEPENDS
  AsterUtilsOpsIncGen

  LINK_LIBS PUBLIC
  AsterInterfaces
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRValueBoundsOpInterface
  MLIRInferTypeOpInterface
  MLIRPtrDialect
  MLIRSideEffectInterfaces
  MLIRTransformUtils
)
