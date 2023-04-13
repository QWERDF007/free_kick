add_subdirectory(utils)
add_subdirectory(cucv)

add_executable(free_kick main.cpp)
target_link_libraries(free_kick PRIVATE
    utils
)