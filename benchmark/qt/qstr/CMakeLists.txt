set(TARGET_NAME free_kick_qstr_benchmark)

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.h *.hpp)

add_executable(${TARGET_NAME} 
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${TARGET_NAME}
PRIVATE
    benchmark::benchmark
    free_kick_qstr free_kick_qstr_headers
)