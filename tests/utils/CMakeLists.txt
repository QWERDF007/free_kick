file(GLOB SRCS *.cpp)


add_executable(utils_test
    ${SRCS}
)

target_link_libraries(utils_test 
PUBLIC
    GTest::gtest_main GTest::gmock
PRIVATE
    free_kick_utils free_kick_utils_headers
)