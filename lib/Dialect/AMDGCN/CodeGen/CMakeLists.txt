add_mlir_library(AsterAMDGCNCodeGen
  CodeGenPatterns.cpp
  Legalizer.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AMDGPUDialect
  AsterCodeGen
  AsterLegalizer
  AsterUtilsDialect
  LSIRDialect
  MLIRIR
  MLIRTransformUtils
)
