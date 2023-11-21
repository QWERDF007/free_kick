project(free_kick_utils_benchmark LANGUAGES CXX)

add_executable(free_kick_utils_benchmark 
    main.cpp 
)

target_link_libraries(free_kick_utils_benchmark
PRIVATE
    benchmark::benchmark
)