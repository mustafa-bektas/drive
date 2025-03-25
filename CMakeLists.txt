cmake_minimum_required(VERSION 3.14)
project(CarGame)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Raylib package (may need to be installed on the system)
find_package(raylib QUIET)
if (NOT raylib_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 4.5.0
    )
    FetchContent_MakeAvailable(raylib)
endif()

# Common source files
set(COMMON_SOURCES
    src/car.cpp
    src/dqn_environment.cpp
    src/dqn_agent.cpp
    src/neural_network.cpp
    src/visualization_helper.cpp
    src/camera.cpp
    src/rendering.cpp
    src/input_handler.cpp
    src/lane_keeping_environment.cpp
    src/lane_keeping_agent.cpp
)

# Add main executable - using trained_model_demo as the main program
add_executable(${PROJECT_NAME} 
    src/trained_model_demo.cpp
    ${COMMON_SOURCES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link against Raylib
target_link_libraries(${PROJECT_NAME} raylib)

# System-specific libraries
if(APPLE)
    target_link_libraries(${PROJECT_NAME} "-framework Cocoa")
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} pthread m dl)
endif()