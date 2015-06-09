cmake_minimum_required(VERSION 2.8)

project(BoostCompute)

set(CMAKE_MODULE_PATH ${BoostCompute_SOURCE_DIR}/cmake)

# find OpenCL
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OPENCL_INCLUDE_DIRS})

# find Boost
find_package(Boost 1.48 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# optional support for c++11
option(BOOST_COMPUTE_USE_CPP11 "Use C++11 features" OFF)
if(NOT MSVC)
  if(${BOOST_COMPUTE_USE_CPP11})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()

# optional support for offline-caching
option(BOOST_COMPUTE_USE_OFFLINE_CACHE "Use offline cache for OpenCL program binaries" OFF)
if(${BOOST_COMPUTE_USE_OFFLINE_CACHE})
  add_definitions(-DBOOST_COMPUTE_USE_OFFLINE_CACHE)
endif()

# thread-safety options
option(BOOST_COMPUTE_THREAD_SAFE "Compile with BOOST_COMPUTE_THREAD_SAFE defined" OFF)
if(${BOOST_COMPUTE_THREAD_SAFE})
  add_definitions(-DBOOST_COMPUTE_THREAD_SAFE)
  if(${BOOST_COMPUTE_USE_CPP11})
    if(MSVC)
      if (MSVC_VERSION GREATER 1800)
        add_definitions(-DBOOST_COMPUTE_HAVE_THREAD_LOCAL)
      endif()
    else()
      add_definitions(-DBOOST_COMPUTE_HAVE_THREAD_LOCAL)
    endif()
  endif()
endif()

# optional third-party libraries
option(BOOST_COMPUTE_HAVE_EIGEN "Have Eigen" OFF)
option(BOOST_COMPUTE_HAVE_OPENCV "Have OpenCV" OFF)
option(BOOST_COMPUTE_HAVE_QT "Have Qt" OFF)
option(BOOST_COMPUTE_HAVE_VTK "Have VTK" OFF)
option(BOOST_COMPUTE_HAVE_CUDA "Have CUDA" OFF)
option(BOOST_COMPUTE_HAVE_TBB "Have TBB" OFF)
option(BOOST_COMPUTE_HAVE_BOLT "Have BOLT" OFF)

include_directories(include)

if(${OpenCL_HEADER_CL_EXT_FOUND})
  add_definitions(-DBOOST_COMPUTE_HAVE_HDR_CL_EXT)
endif()

if(MSVC)
  # optional support for boost dynamic libraries
  option(BOOST_COMPUTE_BOOST_ALL_DYN_LINK "Use boost dynamic link libraries" OFF)
  if(${BOOST_COMPUTE_BOOST_ALL_DYN_LINK})
    add_definitions(-DBOOST_ALL_DYN_LINK)
  endif()
  link_directories(${Boost_LIBRARY_DIRS})
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif()

# compiler options
option(BOOST_COMPUTE_ENABLE_COVERAGE "Enable code coverage generation" OFF)

option(BOOST_COMPUTE_BUILD_TESTS "Build the Boost.Compute tests" OFF)
if(${BOOST_COMPUTE_BUILD_TESTS})
  enable_testing()
  add_subdirectory(test)
endif()

option(BOOST_COMPUTE_BUILD_BENCHMARKS "Build the Boost.Compute benchmarks" OFF)
if(${BOOST_COMPUTE_BUILD_BENCHMARKS})
  add_subdirectory(perf)
endif()

option(BOOST_COMPUTE_BUILD_EXAMPLES "Build the Boost.Compute examples" OFF)
if(${BOOST_COMPUTE_BUILD_EXAMPLES})
  add_subdirectory(example)
endif()

# configure cmake config file
configure_file(
  cmake/BoostComputeConfig.cmake.in
  ${BoostCompute_BINARY_DIR}/BoostComputeConfig.cmake
  @ONLY
)

# install cmake config file
install(
  FILES ${BoostCompute_BINARY_DIR}/BoostComputeConfig.cmake
  DESTINATION lib/cmake/BoostCompute
)

# install header files
install(DIRECTORY include/boost DESTINATION include/compute)
