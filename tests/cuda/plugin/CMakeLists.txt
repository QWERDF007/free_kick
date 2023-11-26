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


if(WIN32)
    
    if(${TensorRT_VERSION_MAJOR} STREQUAL "8")
        set(TensorRT_BINS "${TensorRT_LIBRARY_DIR}/nvinfer_builder_resource.dll")
    else()
        set(TensorRT_BINS)
    endif()
    
    foreach(lib_file ${TensorRT_LIBS})
        string(REPLACE ".lib" ".dll" bin_file ${lib_file})
        list(APPEND TensorRT_BINS ${bin_file})
    endforeach()
    

    # 添加自定义命令, POST_BUILD: 指定目标其他规则执行完后执行, copy: 复制, copy_if_different: 输入有变则复制
    add_custom_command(TARGET tensorrt_plugins_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${TensorRT_BINS}
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
        COMMENT "Copying ${TensorRT_BINS} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
    )
endif()