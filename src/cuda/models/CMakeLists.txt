set(TARGET_NAME tensorrt_model)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

set(MODEL_SOURCES)
set(MODEL_CU_SOURCES)

set(MODEL_LISTS
    rcnn
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

foreach(MODEL_ITER ${MODEL_LISTS})
    include_directories(${MODEL_ITER})
    add_subdirectory(${MODEL_ITER})
endforeach(MODEL_ITER)

# Add common
add_subdirectory(common)
include_directories(.)

set_source_files_properties(${MODEL_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${GENCODES} ${ENABLED_SMS}")
list(APPEND MODEL_SOURCES "${MODEL_CU_SOURCES}")

################################## SHARED LIBRARY #######################################
add_library(${SHARED_TARGET} SHARED
    ${MODEL_SOURCES}
)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)


target_link_libraries(${SHARED_TARGET} PUBLIC
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    # ${MYELIN64_1_LIB}
    CUDA::cudart
    # CUDA::nvrtc
    ${CUDNN_LIB}
)

