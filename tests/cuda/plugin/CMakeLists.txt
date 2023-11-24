set(TARGET_NAME tensorrt_plugins_test)

file(GLOB SRCS *.cpp)


add_executable(${TARGET_NAME}
    ${SRCS}
)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)


target_link_libraries(${TARGET_NAME}
PUBLIC
    GTest::gtest_main
    GTest::gmock
    tensorrt_plugins_headers
PRIVATE
    tensorrt_plugins
)