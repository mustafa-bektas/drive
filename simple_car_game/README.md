# Simple Car Game

A very simple game built with Raylib that features a drivable car (represented as a box) and a ground plane with lane markings.

## Controls

- Arrow Up: Accelerate
- Arrow Down: Brake/Reverse
- Arrow Left: Turn left
- Arrow Right: Turn right

## Building the Project

### Prerequisites

- CMake (version 3.10 or higher)
- A C++ compiler (supporting C++11)
- Git (for fetching Raylib)

### Build Instructions

1. Clone the repository
2. Create a build directory:
   ```
   mkdir build
   cd build
   ```
3. Run CMake:
   ```
   cmake ..
   ```
4. Build the project:
   - On Windows with Visual Studio:
     ```
     cmake --build . --config Release
     ```
   - On Linux/macOS:
     ```
     make
     ```
5. Run the game from the build directory

## Features

- Simple 3D driving mechanics
- Lane markings on the road
- Camera that follows the car
