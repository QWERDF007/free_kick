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
# 指定 pybind11 的 python环境
set(PYTHON_EXECUTABLE "D:/Software/anaconda3/envs/py38/python.exe")
add_subdirectory(pybind11)

# set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind11" PARENT_SCOPE)

# benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing.")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

# sqlpp11
set(BUILD_SQLITE3_CONNECTOR ON)
add_subdirectory(sqlpp11)

# matplotplutplus
add_subdirectory(matplotplusplus)