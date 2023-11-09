file(GLOB SRCS *.cpp)


add_executable(cv_test
    ${SRCS}
)

target_include_directories(cv_test 
PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cv_test 
PUBLIC
    GTest::gtest_main
    GTest::gmock
PRIVATE
    ${OpenCV_LIBS}
)
