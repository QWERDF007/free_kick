# 搭配 `ExternalProject_Add` 创建一个自定义目标来驱动外部项目
# 的下载、更新/补丁、配置、构建、安装和测试步骤
include(ExternalProject)

# 设置 PYPROJ_COMMON_ARGS 变量，该变量包含了一些CMake选项和参数
# CMAKE_INSTALL_PREFIX 指定了安装目录的路径，${CMAKE_CURRENT_BINARY_DIR} 表示当前构建目录的路径。
set(PYPROJ_COMMON_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR} )

# 添加 CMAKE_BUILD_TYPE 到 PYPROJ_COMMON_ARGS
if(CMAKE_BUILD_TYPE)
    list(APPEND PYPROJ_COMMON_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

# 获取名为 nvcv_types 的目标的源代码目录，并将其存储在变量 NVCV_TYPES_SOURCE_DIR 中
# get_target_property(NVCV_TYPES_SOURCE_DIR nvcv_types SOURCE_DIR)

# 添加 CMake 选项和参数到 PYPROJ_COMMON_ARGS
list(APPEND PYPROJ_COMMON_ARGS
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=true                           # 在安装时使用链接路径作为RPATH。RPATH：run-time search path, 可执行或 .so 运行时搜索路径
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=true                                # 在构建时使用源目录作为RPATH。
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}                     # 指定安装库的目录。
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}                     # 指定安装的前缀路径。
    -DCMAKE_MODULE_PATH=${CMAKE_CURRENT_BINARY_DIR}/cmake              # 指定CMake模块的路径
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY} # 指定库的输出目录。
    -DPYBIND11_SOURCE_DIR=${PYBIND11_SOURCE_DIR}                       # 指定pybind11库的源代码目录。   
    # -DDLPACK_SOURCE_DIR=${DLPACK_SOURCE_DIR}                           # 指定dlpack库的源代码目录。
    -DWARNINGS_AS_ERRORS=${WARNINGS_AS_ERRORS}                         # 将警告视为错误。
    -DENABLE_COMPAT_OLD_GLIBC=${ENABLE_COMPAT_OLD_GLIBC}               # 启用兼容旧版本GLIBC。
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}                             # 指定C编译器。
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}                         # 指定C++编译器。
)

# 在交叉编译时覆盖 PYTHON_MODULE_EXTENSION 以生成正确名称的 Python 模块名称
# 例如：set(PYTHON_MODULE_EXTENSION .cpython-py38-aarch64-linux-gnu.so)
if (CMAKE_CROSSCOMPILING)
    list(APPEND PYPROJ_COMMON_ARGS
        -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}
        -DPYTHON_MODULE_EXTENSION=${PYTHON_MODULE_EXTENSION}
    )
endif()

# 每个指定的 Python 版本创建一个名为 free_kick_python${VER} 的外部项目
foreach(VER ${PYTHON_VERSIONS})
    set(BASEDIR ${CMAKE_CURRENT_BINARY_DIR}/python${VER}) 

    ExternalProject_Add(free_kick_python${VER}
        PREFIX ${BASEDIR}                                        # 指定项目的基本目录，即${BASEDIR}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python            # 指定项目的源代码目录
        CMAKE_ARGS ${PYPROJ_COMMON_ARGS} -DPYTHON_VERSION=${VER} # 指定传递给 CMake 的参数
        BINARY_DIR ${BASEDIR}/build                              # 指定项目的构建目录
        TMP_DIR ${BASEDIR}/tmp                                   # 指定项目的临时目录 
        STAMP_DIR ${BASEDIR}/stamp                               # 指定项目的标记目录
        BUILD_ALWAYS true                                        # 始终构建项目
        # DEPENDS nvcv_types cvcuda                                # 指定项目的依赖项
        INSTALL_COMMAND ""                                       # 安装命令, 为空表示不执行任何命令
    )
endforeach()
