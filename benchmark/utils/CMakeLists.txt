set(TARGET_NAME free_kick_utils_benchmark)

project(${TARGET_NAME} LANGUAGES CXX)

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.h *.hpp)

add_executable(${TARGET_NAME} 
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${TARGET_NAME}
PRIVATE
    benchmark::benchmark
)