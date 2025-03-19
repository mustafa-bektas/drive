cmake_minimum_required(VERSION 3.10)
project(CarGame)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
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

# Add executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link against Raylib
target_link_libraries(${PROJECT_NAME} raylib)