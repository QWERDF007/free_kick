# 设置 Debug 生成的可执行文件和库文件的名称后缀
set(CMAKE_DEBUG_POSTFIX "_d")

# CMAKE_BUILD_TYPE 不存在或为空, 则设置为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 指定可执行文件和库文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(GNUInstallDirs)

# 检查 git 子模块是否已经初始化, 否则输出错误信息并终止构建过程
if(EXISTS ${CMAKE_SOURCE_DIR}/.git AND EXISTS ${CMAKE_SOURCE_DIR}/.gitmodules)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/.git/modules)
        message(FATAL_ERROR "git submodules not initialized. Did you forget to run 'git submodule update --init'?")
    endif()
endif()

# 设置动态共享库的版本号和符号可见性，以及优化可执行文件的大小
# 接受两个参数 target version
function(setup_dso target version)
    # 使用正则表达式匹配 version 中的所有数字, 并将结果存储在名为 version_list 的变量中
    string(REGEX MATCHALL "[0-9]+" version_list "${version}")
    # 获取索引为 0 的元素，并将其存储在名为 VERSION_MAJOR 的变量中
    list(GET version_list 0 VERSION_MAJOR)
    list(GET version_list 1 VERSION_MINOR)
    list(GET version_list 2 VERSION_PATCH)

    # 设置目标 target 的版本号和 SO 版本号
    set_target_properties(${target} PROPERTIES
        VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
        SOVERSION "${VERSION_MAJOR}"
    )

    # 优化可执行文件的大小 ==========================

    # 配置链接器, 链接时排除所有库、禁止未定义的符号、使用垃圾回收器和按需链接
    target_link_options(${target} PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,--gc-sections -Wl,--as-needed)
    # 将每个函数和其数据放入单独的链接器部分
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)

    # 链接静态 C/C++ 库 ==========================
    target_link_libraries(${target} PRIVATE
        -static-libstdc++
        -static-libgcc
    )

    # 配置符号可见性 ---------------------------------------------
    set_target_properties(${target} PROPERTIES VISIBILITY_INLINES_HIDDEN on
                                               C_VISIBILITY_PRESET hidden
                                               CXX_VISIBILITY_PRESET hidden
                                               CUDA_VISIBILITY_PRESET hidden)
endfunction()