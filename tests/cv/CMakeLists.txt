file(GLOB SRCS *.cpp)


add_executable(cv_test
    ${SRCS}
)

target_include_directories(cv_test 
PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cv_test 
PUBLIC
    GTest::gtest_main
    GTest::gmock
PRIVATE
    # opencv_world450$<$<CONFIG:Debug>:d>
    ${OpenCV_LIBS}
)

if(WIN32)
    set(OPENCV_WORLD_LIB "${OpenCV_BIN_DIR}/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")

    # 添加自定义命令, POST_BUILD: 指定目标其他规则执行完后执行, copy: 复制, copy_if_different: 输入有变则复制
    add_custom_command(TARGET cv_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${OPENCV_WORLD_LIB}$<$<CONFIG:Debug>:d>.dll"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
    )
endif()

