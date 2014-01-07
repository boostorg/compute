include_directories(../include)

if (NOT ${BOOST_COMPUTE_TIMER_USE_STD_CHRONO})
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} chrono system)
endif()

if (${BOOST_COMPUTE_USE_OFFLINE_CACHE})
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} filesystem)
endif()

if (DEFINED BOOST_COMPONENTS)
  find_package(Boost 1.48 REQUIRED COMPONENTS ${BOOST_COMPONENTS})
  include_directories(${Boost_INCLUDE_DIRS})
endif()

set(BENCHMARKS
  cart_to_polar
  copy_to_device
  count_int
  mersenne_twister
  partial_sum
  saxpy
  sort_float
  sort_uint
  sum_int
)

foreach(BENCHMARK ${BENCHMARKS})
  set(PERF_TARGET perf_${BENCHMARK})
  add_executable(${PERF_TARGET} perf_${BENCHMARK}.cpp)
  target_link_libraries(${PERF_TARGET} ${OPENCL_LIBRARIES})

  if(DEFINED BOOST_COMPONENTS)
    target_link_libraries(${PERF_TARGET} ${Boost_LIBRARIES})
  endif()
endforeach()
