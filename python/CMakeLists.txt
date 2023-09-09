cmake_minimum_required(VERSION 3.15)

project(free_kick_python
        LANGUAGES C CXX
        VERSION 0.0.1
        DESCRIPTION "自由球"
)

# set(CMAKE_CXX_STANDARD 20)

include(../cmake/ConfigCompiler.cmake)

# 设置目录属性, 将这些编译选项应用于目录中的所有源文件的编译过程中。
# ${COMPILE_WARNINGS}是一个变量，它应该包含了一组编译警告选项
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${COMPILE_WARNINGS}")

# 将生成的目标代码设置为位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE true)

# 设置 pybind11 的 python 版本
set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
add_subdirectory(${PYBIND11_SOURCE_DIR} pybind11)

# 寻找 nvcv_types 库
# find_package(nvcv_types REQUIRED)

# 将 PYTHON_MODULE_NAME 设置为 python 加上 PYTHON_VERSION 的值，并且将其中的点号.替换为空字符串
set(PYTHON_MODULE_NAME python${PYTHON_VERSION})
string(REPLACE "." "" PYTHON_MODULE_NAME "${PYTHON_MODULE_NAME}")

include(GNUInstallDirs)
# 定义一个空的缓存变量, 并将其设置为内部缓存, 后续将文件名添加到这个变量中
set(PYTHON_MODULE_FILENAME_LIST "" CACHE INTERNAL "")
function(free_kick_python_add_module)
    # 解析命令行参数。该宏将命令行参数解析为具有指定名称的变量，并将其与给定的参数关联起来。
    # ARG 是解析后的参数结构体
    # SHARED 和 MODULE 是可选的布尔参数
    # TARGET 和 OUTPUT_NAME 是可选的字符串参数
    # SOURCES 是可选的源文件列表参数
    cmake_parse_arguments(ARG "SHARED;MODULE" "TARGET;OUTPUT_NAME" "SOURCES" ${ARGV})

    if(ARG_SHARED AND ARG_MODULE)
        message(FATAL_ERROR "Must specify either SHARED or MODULE")
    elseif(ARG_SHARED)
        set(type "SHARED")
    else()
        set(type "MODULE")
    endif()

    # pybind11 添加模块
    pybind11_add_module(${ARG_TARGET} ${type} ${ARG_SOURCES})

    message(STATUS "--------xxxxxx    ${ARG_TARGET} ${type} ${ARG_SOURCES}")

    # 设置目标的 Debug 后缀为空和库输出目录
    set_target_properties(${ARG_TARGET} PROPERTIES
        DEBUG_POSTFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )

    # 设置目标的输出名称
    if(ARG_OUTPUT_NAME)
        set_target_properties(${ARG_TARGET} PROPERTIES
            OUTPUT_NAME "${ARG_OUTPUT_NAME}"
        )
    else()
        set(ARG_OUTPUT_NAME ${ARG_TARGET})
    endif()

    # 链接静态版本的 stdc++
    target_link_libraries(${ARG_TARGET}
        PRIVATE
            -static-libstdc++
    )

    # 获取目标属性中的前缀和后缀
    get_target_property(prefix ${ARG_TARGET} PREFIX)
    get_target_property(suffix ${ARG_TARGET} SUFFIX)

    # 将目标添加到 PYTHON_MODULE_FILENAME_LIST 变量中
    set(PYTHON_MODULE_FILENAME_LIST
        "${PYTHON_MODULE_FILENAME_LIST};${prefix}${ARG_OUTPUT_NAME}${suffix}" CACHE INTERNAL "")

    # 将 ${ARG_TARGET} 指定的目标（通常是一个 Python 模块）
    # 安装到 ${CMAKE_INSTALL_LIBDIR}/python 目录中，
    # 并将其作为 ${PYTHON_MODULE_NAME} 组件进行安装。
    install(TARGETS ${ARG_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
        COMPONENT ${PYTHON_MODULE_NAME}
)
endfunction()


add_subdirectory(example)

# pybind11 method:
# pybind11_add_module(example example.cpp)

# set_target_properties(example PROPERTIES
#     DEBUG_POSTFIX ""
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
# )