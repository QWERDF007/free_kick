# find_package(cvcuda REQUIRED)

free_kick_python_add_module(
    TARGET free_kick_module_python SHARED
    OUTPUT_NAME free_kick
    SOURCES
        example.cpp
)

# target_link_libraries(free_kick_module_python
#     PRIVATE
#         CUDA::cudart_static
#         pynvcv
#         nvcv_util_compat
#         cvcuda
#         nvcv_python_common
# )

set_target_properties(free_kick_module_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python)