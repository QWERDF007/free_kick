set(TARGET_NAME free_kick_cuda_ops_test)

file(GLOB SRCS *.cpp)


add_executable(${TARGET_NAME}
    ${SRCS}
)



target_link_libraries(${TARGET_NAME} 
PUBLIC
    GTest::gtest_main GTest::gmock
PRIVATE
    free_kick_cuda_ops free_kick_cuda_ops_headers
    opencv_core opencv_imgproc
    CUDA::cudart
)