file(GLOB UTILS_SRCS *.cpp)

add_library(utils STATIC
    ${UTILS_SRCS}
)

target_include_directories(utils PUBLIC .)

add_library(utils_headers INTERFACE)
target_include_directories(utils_headers
    INTERFACE .
)