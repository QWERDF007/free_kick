file(GLOB SRCS *.cpp)


add_executable(qstr_test
    ${SRCS}
)

target_link_libraries(qstr_test 
PUBLIC
    GTest::gtest_main
    GTest::gmock
    qstr_headers
PRIVATE
    qstr qstr_headers
)