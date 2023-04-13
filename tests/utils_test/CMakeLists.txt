file(GLOB SRCS *.cpp)

message(${SRCS})

add_executable(utils_test
    ${SRCS}
)

target_link_libraries(utils_test 
PUBLIC
    GTest::gtest_main
    GTest::gmock
    utils_headers
PRIVATE
    utils
)