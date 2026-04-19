set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(aster-tblgen AsterGen
  DESTINATION "${ASTER_TOOLS_INSTALL_DIR}"
  EXPORT Aster
  Generators.cpp
  OpGen.cpp
  OpGen.h
  main.cpp
)

target_link_libraries(aster-tblgen
  PRIVATE
  ASTERSupport
  MLIRTableGenGenerators
)

mlir_check_all_link_libraries(aster-tblgen)
