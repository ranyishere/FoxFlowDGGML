include_directories(${Boost_INCLUDE_DIRS})
#always start with the minimum required version
cmake_minimum_required(VERSION 3.16)

#set the project name
project(DGGML VERSION 1.0 LANGUAGES CXX)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set the C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17) 

# Force it globally for ALL build configurations
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")

add_compile_options(-O3 -g -fno-omit-frame-pointer -fopenmp)
# add_compile_options(-O3 -fopenmp)

# Add the serial host mode by default
# Currently turning it of would break the code
option(ENABLE_SERIAL_HOST "Enable Serial Host Mode" ON)
if(ENABLE_SERIAL_HOST)
    add_definitions(-DDGGML_ENABLE_HOST_SERIAL)
endif()

# Add debugging option
option(USE_DEBUG "Enter debug mode" ON)
if (USE_DEBUG)
    #add_compile_definitions(DGGML_DEBUG)
endif()

# In the build script simpy do this:
#
# Set the variable SUNDIALS_DIR to the SUNDIALS install directory.
# When using the cmake CLI command, this can be done like so:
#   cmake -D SUNDIALS_DIR=/path/to/sundials/installation

#try to find the sundials package for ODE solving
find_package(SUNDIALS 6.7.0 QUIET)
find_package(OpenMP QUIET)

if(NOT SUNDIALS_FOUND)
    message(STATUS "SUNDIALS 6.7.0 not found, fetching with FetchContent...")
    set(FETCHCONTENT_QUIET FALSE)
    # Include FetchContent module
    include(FetchContent)

    message("Downloading SUNDIALS, this may take a few minutes...")
    # Specify Sundials version and URL
    FetchContent_Declare(
            sundials
            GIT_REPOSITORY https://github.com/LLNL/sundials.git
            GIT_TAG v6.7.0
            GIT_PROGRESS TRUE
    )

    # Fetch and make available
    FetchContent_MakeAvailable(sundials)
endif()

# =============================================================================
# MFEM Integration
# =============================================================================
find_package(mfem QUIET)

if(NOT mfem_FOUND)
    message(STATUS "MFEM not found, fetching with FetchContent...")
    set(FETCHCONTENT_QUIET FALSE)
    # Include FetchContent module (in case SUNDIALS was found locally and this wasn't called)
    include(FetchContent)

    message("Downloading MFEM, this may take a few minutes...")
    FetchContent_Declare(
            mfem
            GIT_REPOSITORY https://github.com/mfem/mfem.git
            GIT_TAG v4.7
            GIT_PROGRESS TRUE
    )

    # Disable MFEM examples, miniapps, and tests so it builds much faster
    set(MFEM_ENABLE_EXAMPLES OFF CACHE BOOL "Disable MFEM examples" FORCE)
    set(MFEM_ENABLE_MINIAPPS OFF CACHE BOOL "Disable MFEM miniapps" FORCE)
    set(MFEM_ENABLE_TESTING OFF CACHE BOOL "Disable MFEM testing" FORCE)

    # Fetch and make available
    FetchContent_MakeAvailable(mfem)
endif()
# =============================================================================

# TODO: see if there is a better way than include_directories command
include_directories(dependencies/YAGL/include)
include_directories(thirdparty/simdjson)
include_directories(src)
include_directories(src/Grids)
include_directories(src/Utlities)
include_directories(src/Listeners)
include_directories(src/Grammar)
include_directories(src/CellComplexes)
include_directories(src/CellComplexes/Utils)
include_directories(src/Solvers)
add_subdirectory(dependencies/YAGL)
add_subdirectory(thirdparty/vtu11)
add_subdirectory(src)
#add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(examples)
