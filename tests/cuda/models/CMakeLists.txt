file(GLOB SRCS *.cpp)


add_executable(tensorrt_model_test
    ${SRCS}
)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)


target_link_libraries(tensorrt_model_test 
PUBLIC
    GTest::gtest_main
    GTest::gmock
    tensorrt_model_headers
PRIVATE
    tensorrt_model
)