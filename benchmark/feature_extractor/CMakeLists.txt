project(feature_extractor_benchmark LANGUAGES CXX)

file(GLOB FILES_TO_COMPILE *.cpp)

add_executable(feature_extractor_benchmark 
    ${FILES_TO_COMPILE} 
)

target_include_directories(feature_extractor_benchmark PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(feature_extractor_benchmark PRIVATE
    benchmark::benchmark
    CUDA::cudart
    ${TensorRT_LIBS}
)