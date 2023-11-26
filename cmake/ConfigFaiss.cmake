set(Faiss_VERSION "1.7.4")
set(Faiss_HOME "E:/DevLib/faiss-gpu-1.7.4")
set(Faiss_LIBRARY_DIR "${Faiss_HOME}/lib")
set(Faiss_INCLUDE_DIRS "${Faiss_HOME}/include")
set(Faiss_BIN_DIR "${Faiss_HOME}/bin")
find_library(Faiss_LIB faiss HINTS ${Faiss_LIBRARY_DIR})
set(Faiss_LIBS)
if(Faiss_LIB)
    list(APPEND Faiss_LIBS ${Faiss_LIB})
endif()