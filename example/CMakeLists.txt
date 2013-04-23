include_directories(../include)

set(EXAMPLES
  copy_data
  hello_world
  find_longest_vector
  list_devices
  monte_carlo
  point_centroid
  sort_vector
  simple_kernel
  time_copy
  transform_sqrt
  vector_addition
)

foreach(EXAMPLE ${EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} ${OPENCL_LIBRARIES})
endforeach()
