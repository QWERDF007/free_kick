set(TARGET_NAME tensorrt_plugins)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    group_normalization_plugin
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

# Add common
# add_subdirectory(common)
include_directories(.)

set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${GENCODES} ${ENABLED_SMS}")
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

################################## SHARED LIBRARY #######################################
add_library(${SHARED_TARGET} SHARED
    ${PLUGIN_SOURCES}
)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)


target_link_libraries(${SHARED_TARGET} PUBLIC
    ${TensorRT_LIBS}
    # ${MYELIN64_1_LIB}
    # CUDA::cudart
    # CUDA::nvrtc
    ${CUDNN_LIB}
    tensorrt_common_headers
    tensorrt_common
)

