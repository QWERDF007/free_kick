set(TARGET_NAME "free_kick_misc_deconstructor")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_executable(${TARGET_NAME} 
    ${SOURCES}
    ${HEADERS}
)
