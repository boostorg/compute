include_directories(../include)

set(PERF_BOOST_COMPONENTS system timer chrono)

if (${BOOST_COMPUTE_USE_OFFLINE_CACHE})
  set(PERF_BOOST_COMPONENTS ${PERF_BOOST_COMPONENTS} filesystem)
endif()

if(${BOOST_COMPUTE_THREAD_SAFE} AND NOT ${BOOST_COMPUTE_USE_CPP11})
  set(PERF_BOOST_COMPONENTS ${PERF_BOOST_COMPONENTS} thread)
endif()

find_package(Boost 1.48 REQUIRED COMPONENTS ${PERF_BOOST_COMPONENTS})
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

set(BENCHMARKS
  accumulate
  bernoulli_distribution
  binary_find
  cart_to_polar
  copy_if
  copy_to_device
  count
  discrete_distribution
  erase_remove
  fill
  find_end
  includes
  inner_product
  is_permutation
  is_sorted
  linear_congruential_engine
  merge
  mersenne_twister
  next_permutation
  nth_element
  partial_sum
  partition
  partition_point
  prev_permutation
  reverse
  rotate
  rotate_copy
  host_sort
  saxpy
  search
  search_n
  set_difference
  set_intersection
  set_symmetric_difference
  set_union
  sort
  sort_by_key
  sort_float
  stable_partition
  uniform_int_distribution
  unique
  unique_copy
  exclusive_scan
  shuffle
)

foreach(BENCHMARK ${BENCHMARKS})
  set(PERF_TARGET perf_${BENCHMARK})
  add_executable(${PERF_TARGET} perf_${BENCHMARK}.cpp)
  target_link_libraries(${PERF_TARGET} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endforeach()

# stl benchmarks (for comparison)
set(STL_BENCHMARKS
  stl_accumulate
  stl_count
  stl_find_end
  stl_includes
  stl_inner_product
  stl_merge
  stl_next_permutation
  stl_partial_sum
  stl_partition
  stl_prev_permutation
  stl_reverse
  stl_rotate
  stl_rotate_copy
  stl_saxpy
  stl_search
  stl_search_n
  stl_set_difference
  stl_set_intersection
  stl_set_symmetric_difference
  stl_set_union
  stl_sort
  stl_stable_partition
  stl_unique
  stl_unique_copy
)

# stl benchmarks which rerquire c++11
if(${BOOST_COMPUTE_USE_CPP11})
  list(APPEND
    STL_BENCHMARKS
    stl_is_permutation
    stl_partition_point
  )
endif()

foreach(BENCHMARK ${STL_BENCHMARKS})
  set(PERF_TARGET perf_${BENCHMARK})
  add_executable(${PERF_TARGET} perf_${BENCHMARK}.cpp)
  target_link_libraries(${PERF_TARGET} ${Boost_LIBRARIES})
endforeach()

# cuda/thrust benchmarks (for comparison)
if(${BOOST_COMPUTE_HAVE_CUDA})
  find_package(CUDA 5.0 REQUIRED)

  set(CUDA_BENCHMARKS
    thrust_accumulate
    thrust_count
    thrust_inner_product
    thrust_partial_sum
    thrust_saxpy
    thrust_sort
    thrust_exclusive_scan
  )

  foreach(BENCHMARK ${CUDA_BENCHMARKS})
    set(PERF_TARGET perf_${BENCHMARK})
    cuda_add_executable(${PERF_TARGET} perf_${BENCHMARK}.cu)
    target_link_libraries(${PERF_TARGET} ${CUDA_LIBRARIES} ${Boost_LIBRARIES})
  endforeach()
endif()

# intel tbb benchmarks (for comparison)
if(${BOOST_COMPUTE_HAVE_TBB})
  find_package(TBB REQUIRED)
  include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

  set(TBB_BENCHMARKS
    tbb_accumulate
    tbb_merge
    tbb_sort
  )

  foreach(BENCHMARK ${TBB_BENCHMARKS})
    set(PERF_TARGET perf_${BENCHMARK})
    add_executable(${PERF_TARGET} perf_${BENCHMARK}.cpp)
    target_link_libraries(${PERF_TARGET} ${TBB_LIBRARIES} ${Boost_LIBRARIES})
  endforeach()
endif()
