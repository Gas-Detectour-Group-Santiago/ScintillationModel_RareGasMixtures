cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(ExportAllLevelsWithPrintGas LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Garfield++
find_package(Garfield REQUIRED)

add_executable(export_all_levels
  printGas.cxx)

target_link_libraries(export_all_levels
  PRIVATE
    Garfield::Garfield
)