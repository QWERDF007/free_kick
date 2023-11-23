file(GLOB SRCS *.cpp)


add_executable(tensorrt_plugin_test
    ${SRCS}
)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)


target_link_libraries(tensorrt_plugin_test 
PUBLIC
    GTest::gtest_main
    GTest::gmock
    tensorrt_plugins_headers
PRIVATE
    tensorrt_plugin
)