cmake_minimum_required(VERSION 3.15)
project(ChessEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ONNX Runtime engine
if(DEFINED ENV{ONNXRUNTIME_PATH})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_PATH})
    
    message(STATUS "ONNX Runtime path: ${ONNXRUNTIME_ROOT}")
    
    # Create executable
    add_executable(chess_engine_onnx 
        chess_engine.cpp
        nnue_evaluator.cpp
    )
    
    # Include directories
    target_include_directories(chess_engine_onnx PRIVATE 
        ${ONNXRUNTIME_ROOT}/include
    )
    
    # Find library
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime libonnxruntime
        PATHS ${ONNXRUNTIME_ROOT}/lib
        REQUIRED
    )
    
    # Link libraries
    target_link_libraries(chess_engine_onnx
        ${ONNXRUNTIME_LIB}
        pthread
        dl
    )
    
    message(STATUS "✓ ONNX Runtime engine configured")
    message(STATUS "  Library: ${ONNXRUNTIME_LIB}")
    
else()
    message(STATUS "ONNXRUNTIME_PATH not set")
    message(STATUS "Run: ./build_onnx.sh")
endif()

# Optimizations
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
