include_directories(../include)

# find Boost.Chrono
if(NOT ${BOOST_COMPUTE_TIMER_USE_STD_CHRONO})
  find_package(Boost 1.47 REQUIRED COMPONENTS chrono system)
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

  if(NOT ${BOOST_COMPUTE_TIMER_USE_STD_CHRONO})
    target_link_libraries(${PERF_TARGET} ${Boost_LIBRARIES})
  endif()
endforeach()
