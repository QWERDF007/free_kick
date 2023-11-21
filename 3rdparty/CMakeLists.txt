set(CMAKE_FOLDER 3rdparty)

# disable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

set(CMAKE_POSITION_INDEPENDENT_CODE on)

# GoogleTest -----------------------------
set(INSTALL_GTEST off)
set(BUILD_GMOCK on)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(BUILD_TESTS)
    add_subdirectory(googletest)
endif()

# pybind11 -----------------------------
set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind11" PARENT_SCOPE)

# benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing.")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()