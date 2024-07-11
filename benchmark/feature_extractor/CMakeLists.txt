set(TARGET_NAME feature_extractor_benchmark)

file(GLOB FILES_TO_COMPILE *.cpp)

add_executable(${TARGET_NAME} 
    ${FILES_TO_COMPILE} 
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME} PRIVATE
    benchmark::benchmark
    CUDA::cudart
    ${TensorRT_LIBS}
)