set(TARGET_NAME "free_kick_crashtest")
set(HEADER_TARGET_NAME "${TARGET_NAME}_headers")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_library(${TARGET_NAME}  SHARED
    ${SOURCES}
    ${HEADERS}
)

add_library(${HEADER_TARGET_NAME} INTERFACE)
target_include_directories(${HEADER_TARGET_NAME}
    INTERFACE .
)