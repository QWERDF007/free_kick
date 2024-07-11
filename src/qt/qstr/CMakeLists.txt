set(TARGET_NAME free_kick_qstr)

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.h *.hpp)

qt_add_library(${TARGET_NAME} STATIC
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(${TARGET_NAME} PUBLIC .)
target_link_libraries(${TARGET_NAME} 
PUBLIC
    Qt6::Core
)

set(HEADER_NAME ${TARGET_NAME}_headers)

add_library(${HEADER_NAME} INTERFACE)
target_include_directories(${HEADER_NAME}
    INTERFACE .
)
