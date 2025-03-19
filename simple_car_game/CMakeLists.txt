cmake_minimum_required(VERSION 3.10)
project(SimpleCarGame)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent for downloading raylib
include(FetchContent)

# Fetch and build raylib
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 4.5.0
)

# Make RAYLIB_BUILD_EXAMPLES option OFF
set(RAYLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Make sure raylib is built
FetchContent_MakeAvailable(raylib)

# Add executable
add_executable(${PROJECT_NAME} 
    src/main.cpp
)

# Link raylib
target_link_libraries(${PROJECT_NAME} raylib)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${raylib_SOURCE_DIR}/src)

# Copy resources to build directory
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#     ${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:${PROJECT_NAME}>/resources
# )
