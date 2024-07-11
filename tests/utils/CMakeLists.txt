set(TARGET_NAME free_kick_utils_test)
file(GLOB SRCS *.cpp)


add_executable(${TARGET_NAME}
    ${SRCS}
)

target_link_libraries(${TARGET_NAME} 
PRIVATE
    GTest::gtest_main GTest::gmock
    free_kick_utils free_kick_utils_headers
)