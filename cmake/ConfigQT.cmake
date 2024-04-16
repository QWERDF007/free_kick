# 设置 Qt 的 cmake 文件目录
set(QT_CMAKE_DIR "D:/Software/Qt6/6.5.1/msvc2019_64/lib/cmake/Qt6")
set(QT_QML_IMPORT_DIR "D:/Software/Qt6/6.5.1/msvc2019_64/qml")


# 寻找Qt6的Quick组件
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Quick Widgets Charts PATHS ${QT_CMAKE_DIR})

# 寻找Qt6的测试组件
find_package(Qt6 6.5 REQUIRED COMPONENTS QuickTest)

# 设置Qt6项目
qt_standard_project_setup(REQUIRES 6.5)

# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTORCC ON)
# set(CMAKE_AUTOUIC ON)

# 添加 qml 导入目录
# QML_IMPORT_PATH 用于支持语法高亮
list(APPEND QML_IMPORT_PATH ${CMAKE_BINARY_DIR})
list(APPEND QML_IMPORT_PATH ${QT_QML_IMPORT_DIR})
# 去除重复
list(REMOVE_DUPLICATES QML_IMPORT_PATH)

# 将导入目录添加到 cache, 强制更新
set(QML_IMPORT_PATH ${QML_IMPORT_PATH}
    CACHE STRING "qml import paths"
    FORCE
)