
# 获取所有CPP文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME} PUBLIC
    ${CUDAToolkit_LIBRARY_DIR}
    ${TensorRT_LIBRARY_DIR}
    ${OpenCV_LIBRARY_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        CUDA::cudart
        nvinfer
)