set(OpenCV_HOME "H:/Software/opencv/build/x64/vc16")
set(OpenCV_DIR "${OpenCV_HOME}/lib") # dir contain .cmake
set(OpenCV_LIBRARY_DIR ${OpenCV_DIR})
set(OpenCV_BIN_DIR "${OpenCV_HOME}/bin")
find_package(OpenCV REQUIRED) 
