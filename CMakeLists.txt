cmake_minimum_required(VERSION 2.8)

project(BoostCompute)

set(CMAKE_MODULE_PATH ${BoostCompute_SOURCE_DIR}/cmake)

# find OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

# find Boost
find_package(Boost 1.48 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(include)

if(${OpenCL_HEADER_CL_EXT_FOUND})
  add_definitions(-DBOOST_COMPUTE_HAVE_HDR_CL_EXT)
endif()

option(BOOST_COMPUTE_TIMER_USE_STD_CHRONO "Use std::chrono for timer" OFF)
mark_as_advanced(BOOST_COMPUTE_TIMER_USE_STD_CHRONO)
if(${BOOST_COMPUTE_TIMER_USE_STD_CHRONO})
  add_definitions(-DBOOST_COMPUTE_TIMER_USE_STD_CHRONO)
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif()

option(BOOST_COMPUTE_BUILD_EXAMPLES "Build the Boost.Compute examples" OFF)
if(${BOOST_COMPUTE_BUILD_EXAMPLES})
  add_subdirectory(example)
endif()

option(BOOST_COMPUTE_BUILD_TESTS "Build the Boost.Compute tests" OFF)
if(${BOOST_COMPUTE_BUILD_TESTS})
  enable_testing()
  add_subdirectory(test)
endif()

option(BOOST_COMPUTE_BUILD_BENCHMARKS "Build the Boost.Compute benchmarks" OFF)
if(${BOOST_COMPUTE_BUILD_BENCHMARKS})
  add_subdirectory(perf)
endif()
