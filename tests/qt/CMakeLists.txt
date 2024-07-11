set(TARGET_NAME free_kick_qstr_test)
file(GLOB SRCS *.cpp)


add_executable(${TARGET_NAME}
    ${SRCS}
)

target_link_libraries(${TARGET_NAME} 
PUBLIC
    GTest::gtest_main GTest::gmock
PRIVATE
    free_kick_qstr free_kick_qstr_headers
)