set(TARGET_NAME faiss_sample)

file(GLOB SRCS *.cpp)


add_executable(${TARGET_NAME}
    ${SRCS}
)

include_directories(
    ${Faiss_INCLUDE_DIRS}
)
target_link_libraries(${TARGET_NAME}
PRIVATE
    ${Faiss_LIBS}
)



if(WIN32)
    set(Faiss_BINS "${MKL_BIN_DIR}/mkl_intel_thread.2.dll")
    foreach(lib_file ${Faiss_LIBS})
        string(REPLACE "lib/" "bin/" tmp ${lib_file})
        string(REPLACE ".lib" ".dll" bin_file ${tmp})
        list(APPEND Faiss_BINS ${bin_file})
    endforeach()
    
    # 添加自定义命令, POST_BUILD: 指定目标其他规则执行完后执行, copy: 复制, copy_if_different: 输入有变则复制
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${Faiss_BINS}
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
        COMMENT "Copying ${Faiss_BINS} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
    )
endif()