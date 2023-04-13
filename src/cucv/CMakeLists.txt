# TensorRT plugin
add_subdirectory(plugin)

# header
add_library(cucv_headers INTERFACE)
target_include_directories(cucv_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)

# file(GLOB SRCS *.cpp)
# file(GLOB CU_SRCS priv/*.cu)

# add_library(cucv SHARED
#     ${SRCS}
#     ${CU_SRCS}
# )

# target_include_directories(cucv PUBLIC .)
# target_link_libraries(cucv
# PUBLIC
#     cucv_plugin
# )

