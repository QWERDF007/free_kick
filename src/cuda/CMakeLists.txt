# TensorRT plugin
add_subdirectory(plugin)

# header
add_library(tensorrt_plugin_headers INTERFACE)
target_include_directories(tensorrt_plugin_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)


# TensorRT models
add_subdirectory(model)

# header
add_library(tensorrt_model_headers INTERFACE)
target_include_directories(tensorrt_model_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
)
