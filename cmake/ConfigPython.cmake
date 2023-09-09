list(APPEND PYTHON_VERSIONS 3.9)
# set(PYBIND11_PYTHON_VERSION 3.8 CACHE STRING "")
# set(PYTHON_EXECUTABLE "D:/Software/anaconda3/envs/py38/python.exe" CACHE STRING "")


# 生成 python 库作为子项目, 创建一个虚拟的 Findnvcv.cmake 文件, 
# 以便在项目中使用 find_package 命令来查找和使用 nvcv_types 库。
# 创建名为 nvcv_types 的共享库, 设置其导入属性的接口包含目录。

# set(FINDNVCV_TYPES_CONTENTS
# [=[
# add_library(nvcv_types SHARED IMPORTED)
# target_include_directories(nvcv_types
#     INTERFACE
#     "$<TARGET_PROPERTY:nvcv_types,INTERFACE_INCLUDE_DIRECTORIES>"
# )
# add_library(nvcv_types_headers INTERFACE IMPORTED)
# target_include_directories(nvcv_types_headers
#     INTERFACE
#     "$<TARGET_PROPERTY:nvcv_types,INTERFACE_INCLUDE_DIRECTORIES>"
# )
# ]=])

# 创建一个虚拟的 Findcvcuda.cmake 文件，
# 以便在项目中使用 find_package 命令来查找和使用 cvcuda 库。
# 定义了一个名为 cvcuda 的共享库，并设置了其导入属性。还指定了 cvcuda 库的接口包含目录
# set(FINDCVCUDA_CONTENTS
# [=[
# add_library(cvcuda SHARED IMPORTED)
# target_include_directories(cvcuda
#     INTERFACE
#     "$<TARGET_PROPERTY:cvcuda,INTERFACE_INCLUDE_DIRECTORIES>"
# )
# ]=])

# 根据是否存在多个构建配置类型设置 NVCV_CONFIG_TYPES 为 Debug、Release 等
# if(CMAKE_CONFIGURATION_TYPES)
#     set(NVCV_CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES})
# else()
#     set(NVCV_CONFIG_TYPES ${CMAKE_BUILD_TYPE})
# endif()

# 将每个配置类型转换为小写，并将其用作字符串拼接的一部分
# foreach(cfg ${NVCV_CONFIG_TYPES})
#     string(TOLOWER ${cfg} cfg_lower)
#     set(FINDNVCV_TYPES_CONTENTS
# "${FINDNVCV_TYPES_CONTENTS}include(nvcv_types_${cfg_lower})
# ")
#     set(FINDCVCUDA_CONTENTS
# "${FINDCVCUDA_CONTENTS}include(cvcuda_${cfg_lower})
# ")
# endforeach()

# 生成两个文件：Findnvcv_types.cmake和Findcvcuda.cmake。
# 这些文件用于在CMake项目中查找和配置nvcv_types和cvcuda库的位置和属性。
# file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/Findnvcv_types.cmake CONTENT "${FINDNVCV_TYPES_CONTENTS}")
# file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/Findcvcuda.cmake CONTENT "${FINDCVCUDA_CONTENTS}")

# 根据 NVCV_CONFIG_TYPES 的长度来确定构建后缀
# list(LENGTH "${NVCV_CONFIG_TYPES}" num_configs)
# if(${num_configs} EQUAL 1)
#     set(NVCV_BUILD_SUFFIX "")
# else()
#     set(NVCV_BUILD_SUFFIX "_$<UPPER_CASE:$<CONFIG>>")
# endif()

# 生成一个名为nvcv_types_$<LOWER_CASE:$<CONFIG>>.cmake的文件，并设置nvcv_types目标的属性
# 设置 IMPORTED_LOCATION和IMPORTED_IMPLIB 属性，这些属性指定了 nvcv_types 目标的导入位置和导入库文件。
# file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/nvcv_types_$<LOWER_CASE:$<CONFIG>>.cmake CONTENT
# "set_target_properties(nvcv_types PROPERTIES IMPORTED_LOCATION${NVCV_BUILD_SUFFIX} \"$<TARGET_FILE:nvcv_types>\"
#                                        IMPORTED_IMPLIB${NVCV_BUILD_SUFFIX} \"$<TARGET_LINKER_FILE:nvcv_types>\")
# ")

# 生成一个名为cvcuda_$<LOWER_CASE:$<CONFIG>>.cmake的文件，并设置 cvcuda 目标的属性
# 设置 IMPORTED_LOCATION和IMPORTED_IMPLIB 属性，这些属性指定了 cvcuda 目标的导入位置和导入库文件。
# file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake/cvcuda_$<LOWER_CASE:$<CONFIG>>.cmake CONTENT
# "set_target_properties(cvcuda PROPERTIES IMPORTED_LOCATION${NVCV_BUILD_SUFFIX} \"$<TARGET_FILE:cvcuda>\"
#                                                  IMPORTED_IMPLIB${NVCV_BUILD_SUFFIX} \"$<TARGET_LINKER_FILE:cvcuda>\")
# ")



# 检查是否已经设置了要构建的Python版本
if(PYTHON_VERSIONS)
    set(USE_DEFAULT_PYTHON false)
# 如果没有设置 PYTHON_VERSIONS 变量，则使用 find_package 命令查找 Python 解释器，
# 并将 PYTHON_VERSIONS 设置为找到的 Python 版的主版本号和次版本号的组合
else()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(PYTHON_VERSIONS ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
    set(USE_DEFAULT_PYTHON true)
endif()
