cmake_minimum_required(VERSION 3.26)
project(center_pos VERSION 1.0.1)

set(CMAKE_CXX_STANDARD 14)

set(RESOURCE_FILES center_pos.rc)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

set(OPENCV_INCLUDES1 D:\\Software\\opencv\\build\\include)
set(OPENCV_INCLUDES2 D:\\Software\\opencv\\build\\include\\opencv2)

SET(OPENCV_LIB D:\\Software\\opencv\\build\\x64\\vc16\\lib)
SET(OPENCV_BIN D:\\Software\\opencv\\build\\x64\\vc16\\bin)

include_directories(./include)
include_directories(${OPENCV_INCLUDES1})
include_directories(${OPENCV_INCLUDES2})

link_directories(OPENCV_LIB)

add_library(center_pos SHARED
        visionprocess.cpp
        visionprocessapi.cpp
        logger.cpp
        include/logger.h
        ${RESOURCE_FILES}
)

#add_executable(center_pos
#        main.cpp
#        visionprocess.cpp
#        visionprocessapi.cpp
#
#)

set_target_properties(center_pos PROPERTIES
        VS_RESOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/center_pos.rc"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries (${PROJECT_NAME} ${OPENCV_LIB}\\opencv_world470d.lib)
else()
    target_link_libraries (${PROJECT_NAME} ${OPENCV_LIB}\\opencv_world470.lib)
endif()



