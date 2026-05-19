add_mlir_dialect_library(AMDGPUDialect
  AMDGPU.cpp
  AMDGPUAttrs.cpp
  AMDGPUOps.cpp
  IntRangeImpl.cpp

  DEPENDS
  AMDGPUEnumsIncGen
  AMDGPUOpsIncGen

  LINK_LIBS PUBLIC
  AsterInterfaces
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRSideEffectInterfaces
)
