set(TRAGET_NAME "free_kick_common")
set(HEADER_TARGET_NAME "${TRAGET_NAME}_headers")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

add_library(${TRAGET_NAME} STATIC
    ${SOURCES}
    ${HEADERS}
)
if (WIN32)
    target_link_libraries(${TRAGET_NAME} PRIVATE Dbghelp)
else()
    target_link_libraries(${TRAGET_NAME} PRIVATE dl)
endif()

add_library(${HEADER_TARGET_NAME} INTERFACE)
target_include_directories(${HEADER_TARGET_NAME}
    INTERFACE .
)

