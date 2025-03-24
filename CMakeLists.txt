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

# Common sources
set(COMMON_SOURCES
    src/car.cpp
    src/camera.cpp
    src/rendering.cpp
    src/input_handler.cpp
    src/rl_agent.cpp
    src/episode_manager.cpp
    src/enhanced_episode_manager.cpp  # Add this file
)

# Main game executable
add_executable(CarGame src/main.cpp ${COMMON_SOURCES})
target_include_directories(CarGame PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(CarGame raylib)

# Parallel game executable (separate from main)
add_executable(CarGameParallel 
    src/main_parallel.cpp 
    src/parallel_rl_agent.cpp
    ${COMMON_SOURCES}
)
target_include_directories(CarGameParallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(CarGameParallel raylib)

# Coordinator executable (standalone)
add_executable(parallel_coordinator src/parallel_coordinator.cpp)
target_include_directories(parallel_coordinator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)