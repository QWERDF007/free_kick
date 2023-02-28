message(STATUS "Config SUB_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}")

# 执行命令 ls 当前目录，将输出添加到 SUB_DIR_FILES 列表
# 方便在添加新文件后不用改动
execute_process(
    COMMAND ls ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SUB_DIR_FILES
)

# 将换行符 '\n' 替换成 ';'
string(REPLACE "\n" ";" SUB_DIR_FILES ${SUB_DIR_FILES})

# 遍历当前目录下的子目录
foreach(item ${SUB_DIR_FILES})
    # 把子目录的 src 文件添加到 SUB_DIR_SRCS 并把子目录添加到 SUB_INC_DIRS
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${item})
        # 搜集 ${item} 目录下的源码文件添加到 ${item}_SRC 列表
        aux_source_directory(${item} ${item}_SRC)

        message(STATUS "add '${CMAKE_CURRENT_SOURCE_DIR}/${${item}_SRC}' to free_kick")

        # 添加文件到 SUB_DIR_SRCS 列表
        list(APPEND SUB_DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${${item}_SRC})

        # 添加目录到 SUB_INC_DIRS 列表
        list(APPEND SUB_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_CURRENT_SOURCE_DIR}/${item})
    endif()
endforeach()

# 搜索头文件，添加到 SUB_DIR_INCS 列表
# 然后添加到sources里面，方便在 visual studio 查看
file(GLOB SUB_DIR_INCS *.h *.hpp */*.h */*.hpp)

# 添加源码文件
target_sources(free_kick
    PRIVATE
    ${SUB_DIR_SRCS}
    # ${SUB_DIR_INCS}
)

# # 添加包含目录
# target_include_directories(preprocessor
#     PRIVATE
#     ${SUB_INC_DIRS}
# )

# 添加源码文件到 test
target_sources(free_kick_test
    PRIVATE
    ${SUB_DIR_SRCS}
)