cmake_minimum_required(VERSION 3.15)

project(free_kick
        LANGUAGES C CXX
        VERSION 0.0.1
        DESCRIPTION "自由球"
)

enable_language(CUDA)


# Options to configure the build tree =======
option(BUILD_SAMPLES "Enable samples" OFF)
option(BUILD_TESTS "Enable testsuite" ON)
option(BUILD_DOCS "Enable docs" OFF)
option(ENABLE_SANITIZER "Enabled sanitized build" OFF)
option(BUILD_PYTHON "Build python bindings" ON)

# Configure build tree ======================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

include(ConfigBuildTree)
include(ConfigCompiler)
include(ConfigCUDA)
include(ConfigTensorRT)

# Define the build tree ====================
add_subdirectory(3rdparty EXCLUDE_FROM_ALL)

add_subdirectory(src)

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()


message(STATUS "")
include(PrintConfig)