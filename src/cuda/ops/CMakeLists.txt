set(TARGET_NAME free_kick_cuda_ops)

file(GLOB SOURCES *.cpp *.cu)
file(GLOB HEADERS *.h *.hpp)

add_library(${TARGET_NAME} STATIC
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(${TARGET_NAME} PUBLIC .)
target_link_libraries(${TARGET_NAME} 
PUBLIC
    CUDA::cudart
    opencv_world
)

# 将名称转换为大写
string(TOUPPER ${TARGET_NAME} TARGET_NAME_UPPER)
# 设置额外编译定义
set(TARGET_EXPORTS "${TARGET_NAME_UPPER}_LIBRARY")
# 添加一个宏定义, 配合动态库导出函数、类
# https://stackoverflow.com/a/67923443
target_compile_definitions(${TARGET_NAME} PRIVATE ${TARGET_EXPORTS})

# 添加接口头文件，链接目标后可以include, 无需另外包含头文件目录
set(HEADER_NAME ${TARGET_NAME}_headers)
add_library(${HEADER_NAME} INTERFACE)
target_include_directories(${HEADER_NAME}
    INTERFACE .
)
