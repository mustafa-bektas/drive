cmake_minimum_required(VERSION 3.28)
project(CarGame)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
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

# Common source files (used by both graphical and headless versions)
set(COMMON_SOURCES
    src/car.cpp
    src/dqn_environment.cpp
    src/dqn_agent.cpp
    src/neural_network.cpp
)

# Graphical version source files
set(GRAPHICAL_SOURCES
    src/camera.cpp
    src/rendering.cpp
    src/input_handler.cpp
    src/dqn_visualizer.cpp
)

add_executable(CarGameDemo src/trained_model_demo.cpp ${COMMON_SOURCES} ${GRAPHICAL_SOURCES})
target_include_directories(CarGameDemo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(CarGameDemo raylib)

