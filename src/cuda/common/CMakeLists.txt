# PARENT_SCOPE选项确保该变量在此目录之外可见
# file(GLOB SRCS *.cpp)
# set(PLUGIN_SOURCES ${PLUGIN_SOURCES} ${SRCS})
# set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)
# file(GLOB CU_SRCS *.cu)
# set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} ${CU_SRCS})
# set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} PARENT_SCOPE)

set(TARGET_NAME tensorrt_common)

file(GLOB COMMON_SRCS *.cpp)

add_library(${TARGET_NAME} STATIC
    ${COMMON_SRCS}
)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME} PUBLIC
    CUDA::cudart
    ${NVINFER_LIB}
)