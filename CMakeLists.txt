CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(MKLML REQUIRED)
IF(MKLML_FOUND)
  IF (NOT WIN32)
    SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration ${CMAKE_C_FLAGS}")
  ENDIF (NOT WIN32)
   
  SET(src init.c tensor.c)
  SET(luasrc init.lua tensor.lua FFI.lua)
  
  ADD_TORCH_PACKAGE(mkltorch "${src}" "${luasrc}")
  TARGET_LINK_LIBRARIES(mkltorch luaT ${MKLML_LIBRARIES})
  
  IF(LUALIB)
    TARGET_LINK_LIBRARIES(mkltorch ${LUALIB})
  ENDIF()
  
  SET(MKLTorch_INSTALL_CMAKE_SUBDIR "share/cmake/mkltorch" CACHE PATH
      "Install dir for .cmake files (relative to MKLTorch_INSTALL_PREFIX)")
  SET(MKLTorch_INSTALL_INCLUDE_SUBDIR "include/mkltorch" CACHE PATH
      "Install dir for  header files (relative to MKLTorch_INSTALL_PREFIX)")
  SET(MKLTorch_INSTALL_LIB_SUBDIR "lib" CACHE PATH
      "Install dir for libraries files (relative to MKLTorch_INSTALL_PREFIX)")

  SET(MKLTorch_INSTALL_PREFIX "${Torch_INSTALL_PREFIX}")
  SET(MKLTorch_INSTALL_LIB "${MKLTorch_INSTALL_PREFIX}/${MKLTorch_INSTALL_LIB_SUBDIR}")
  SET(MKLTorch_INSTALL_INCLUDE "${MKLTorch_INSTALL_PREFIX}/${MKLTorch_INSTALL_INCLUDE_SUBDIR}")
  SET(MKLTorch_INSTALL_CMAKE "${MKLTorch_INSTALL_PREFIX}/${MKLTorch_INSTALL_CMAKE_SUBDIR}")

  CONFIGURE_FILE("cmake/MKLTorchConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/MKLTorchConfig.cmake" @ONLY)

  SET(MKLTorch_LIBRARIES mkltorch)
  message("${MKLTorch_INSTALL_LIB}  ${MKLTorch_LIBRARIES}   ${MKLTorch_INSTALL_INCLUDE} ${MKLTorch_INSTALL_LIB}")

  INSTALL(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/MKLTorchConfig.cmake"
    DESTINATION "${MKLTorch_INSTALL_CMAKE}")

  INSTALL(TARGETS mkltorch 
          LIBRARY DESTINATION "${MKLTorch_INSTALL_LIB}")
  
  INSTALL(FILES
          init.c
          tensor.c
          MKLTensor.h
          DESTINATION "${MKLTorch_INSTALL_INCLUDE}")
  
  INSTALL(FILES
          generic/tensor.h
          generic/tensor.c
          DESTINATION "${MKLTorch_INSTALL_INCLUDE}/generic")

ELSE(MKLML_FOUND)
  

  MESSAGE(FATAL_ERROR "MKLML libraries is not found, compilation is terminated, please refer to installation manul")

ENDIF(MKLML_FOUND)




