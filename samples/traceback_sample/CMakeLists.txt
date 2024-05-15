set(TARGET_NAME "traceback_sample")

add_subdirectory(crashtest)

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_executable(${TARGET_NAME} 
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${TARGET_NAME} 
PRIVATE
    free_kick_common free_kick_common_headers
    crashtest crashtest_headers
)