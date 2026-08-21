cmake_minimum_required(VERSION 3.16)
project(stereo_mapping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags for Raspberry Pi 5
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=cortex-a76 -mtune=cortex-a76")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc imgcodecs calib3d highgui
)

# Check for ximgproc (WLS filter)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.0")
    list(APPEND OpenCV_REQUIRED_COMPONENTS ximgproc)
    find_package(OpenCV REQUIRED COMPONENTS ximgproc)
    set(HAVE_XIMGPROC TRUE)
else()
    set(HAVE_XIMGPROC FALSE)
    message(WARNING "OpenCV < 4.0: WLS filter not available")
endif()

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Stereo pipeline library
add_library(stereo_pipeline_lib STATIC
    src/stereo_pipeline.cpp
)
target_link_libraries(stereo_pipeline_lib
    ${OpenCV_LIBS}
    yaml-cpp
)

# Main stereo pipeline executable
add_executable(stereo_pipeline
    src/main.cpp
)
target_link_libraries(stereo_pipeline
    stereo_pipeline_lib
    ${OpenCV_LIBS}
    yaml-cpp
)

# Install
install(TARGETS stereo_pipeline
    RUNTIME DESTINATION bin
)

message(STATUS "")
message(STATUS "=== Stereo Mapping Build Configuration ===")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  WLS filter: ${HAVE_XIMGPROC}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "============================================")
message(STATUS "")
