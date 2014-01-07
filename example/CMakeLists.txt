include_directories(../include)

set(EXAMPLES
  black_scholes
  copy_data
  hello_world
  longest_vector
  list_devices
  monte_carlo
  point_centroid
  price_cross
  sort_vector
  simple_kernel
  time_copy
  transform_sqrt
  vector_addition
)

if (${BOOST_COMPUTE_USE_OFFLINE_CACHE})
  find_package(Boost 1.46 REQUIRED COMPONENTS system filesystem)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

foreach(EXAMPLE ${EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endforeach()
