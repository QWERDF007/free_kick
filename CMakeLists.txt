cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

project(free_kick)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE SRCS "cc/*.cpp" "cc/*.h")


# googletest
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/googletest)

if(WIN32)
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

add_executable(free_kick ${PROJECT_SOURCE_DIR}/cc/main.cpp)
add_executable(free_kick_test)

add_subdirectory(${PROJECT_SOURCE_DIR}/cc)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)


target_include_directories(free_kick
    PUBLIC
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/googletest/include
)

# # 添加链接目录
# target_link_directories(free_kick
    
# )

target_include_directories(free_kick_test
    PUBLIC
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/googletest/include
    ${PROJECT_SOURCE_DIR}/cc
)

target_link_libraries(free_kick_test
    gtest_main
)

