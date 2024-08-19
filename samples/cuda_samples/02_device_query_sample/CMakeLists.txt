set(TARGET_NAME "free_kick_device_query_sample")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_executable(${TARGET_NAME} 
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${TARGET_NAME}  
PRIVATE 
    CUDA::cudart
    # free_kick_common free_kick_common_headers
)