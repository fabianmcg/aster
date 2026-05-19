add_mlir_library(ASTERAnalysis
  DPSAnalysis.cpp
  LivenessAnalysis.cpp
  RegisterConstraints.cpp
  ThreadUniformAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AMDGPUDialect
  ASTERIR
  ASTERSupport
  AsterInterfaces
  AsterUtilsDialect
  LSIRDialect
  MLIRAnalysis
  MLIRGPUDialect
  MLIRIR
  MLIRSupport
)
