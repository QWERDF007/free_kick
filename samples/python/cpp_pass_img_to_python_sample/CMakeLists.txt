
set(TARGET_NAME "free_kick_cpp_pass_img_to_python_sample")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# 获取所有头文件的相对路径
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)



add_executable(${TARGET_NAME} 
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${TARGET_NAME}  
PRIVATE 
    pybind11::embed
    free_kick_common free_kick_common_headers
    ${OpenCV_LIBS}
)


target_include_directories(${TARGET_NAME} 
PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)