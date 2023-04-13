file(GLOB SRCS *.cpp)

message(${SRCS})

add_executable(cucv_test
    ${SRCS}
)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)


target_link_libraries(cucv_test 
PUBLIC
    GTest::gtest_main
    GTest::gmock
    cucv_headers
PRIVATE
    cucv_plugin
)