set(PROJECT_NAME free_kick_utils_benchmark)

project(${PROJECT_NAME} LANGUAGES CXX)

file(GLOB SRCS *.cpp)

add_executable(${PROJECT_NAME} 
    ${SRCS}
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
    benchmark::benchmark
)