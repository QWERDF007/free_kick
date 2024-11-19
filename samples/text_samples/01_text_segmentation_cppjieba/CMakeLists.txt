set(TARGET_NAME "free_kick_text_segmentation_cppjieba")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_executable(${TARGET_NAME} 
    ${SOURCES}
    ${HEADERS}
)

message(STATUS "${CMAKE_SOURCE_DIR}/3rdparty/cppjieba/include")
target_include_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_SOURCE_DIR}/3rdparty/cppjieba/include
    ${CMAKE_SOURCE_DIR}/3rdparty/cppjieba/deps/limonp/include
)
