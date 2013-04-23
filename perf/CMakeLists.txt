include_directories(../include)

set(BENCHMARKS
  blas_saxpy
  copy_to_device
  count_int
  partial_sum
  sort_float
  sort_uint
  sum_int
)

foreach(BENCHMARK ${BENCHMARKS})
  set(PERF_TARGET perf_${BENCHMARK})
  add_executable(${PERF_TARGET} perf_${BENCHMARK}.cpp)
  target_link_libraries(${PERF_TARGET} ${OPENCL_LIBRARIES})
endforeach()
