# common lib
add_subdirectory(common)

add_library(tensorrt_common_headers INTERFACE)
target_include_directories(tensorrt_common_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)



# TensorRT plugin
add_subdirectory(plugins)

# header
add_library(tensorrt_plugins_headers INTERFACE)
target_include_directories(tensorrt_plugins_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins
)


# TensorRT models
add_subdirectory(models)

# header
add_library(tensorrt_models_headers INTERFACE)
target_include_directories(tensorrt_models_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)
