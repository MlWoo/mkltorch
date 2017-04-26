CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

#SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/THC/cmake ${CMAKE_MODULE_PATH})

FIND_PACKAGE(Torch REQUIRED)

IF (NOT WIN32)
SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration ${CMAKE_C_FLAGS}")
ENDIF (NOT WIN32)


SET(src init.c tensor.c)
SET(luasrc init.lua tensor.lua FFI.lua)

ADD_TORCH_PACKAGE(mkltorch "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(mkltorch luaT)

IF(LUALIB)
  TARGET_LINK_LIBRARIES(mkltorch ${LUALIB})
ENDIF()

INSTALL(TARGETS mkltorch 
        LIBRARY DESTINATION "${Torch_INSTALL_LIB}")

INSTALL(FILES
        init.c
        tensor.c
        MKLTensor.h
        DESTINATION "${Torch_INSTALL_INCLUDE}/mkltorch")



INSTALL(FILES
        generic/tensor.h
        generic/tensor.c
        DESTINATION "${Torch_INSTALL_INCLUDE}/mkltorch/generic")



