include_directories(../include)

set(EXAMPLES
  amd_cpp_kernel
  black_scholes
  copy_data
  fizz_buzz
  hello_world
  host_sort
  inline_ptx
  longest_vector
  list_devices
  mapped_view
  memory_limits
  monte_carlo
  point_centroid
  price_cross
  print_vector
  sort_vector
  simple_kernel
  time_copy
  transform_sqrt
  vector_addition
  simple_moving_average
  matrix_transpose
  matrix_multiplication
)

# boost library link dependencies
set(EXAMPLE_BOOST_COMPONENTS program_options)
if (${BOOST_COMPUTE_USE_OFFLINE_CACHE})
  set(EXAMPLE_BOOST_COMPONENTS ${EXAMPLE_BOOST_COMPONENTS} system filesystem)
endif()

find_package(Boost 1.48 REQUIRED COMPONENTS ${EXAMPLE_BOOST_COMPONENTS})
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

foreach(EXAMPLE ${EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endforeach()

# eigen examples
if(${BOOST_COMPUTE_HAVE_EIGEN})
  find_package(Eigen REQUIRED)
  include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
  add_executable(batched_determinant batched_determinant.cpp)
  target_link_libraries(batched_determinant ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endif()

# opencv examples
if(${BOOST_COMPUTE_HAVE_OPENCV})
  find_package(OpenCV REQUIRED)
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

  set(OPENCV_EXAMPLES
    k_means
    opencv_filter
    random_walk
    opencv_optical_flow 
  )

  foreach(EXAMPLE ${OPENCV_EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
  endforeach()
endif()

# opengl/vtk examples
if(${BOOST_COMPUTE_HAVE_VTK})
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
  add_executable(opengl_sphere opengl_sphere.cpp)
  target_link_libraries(opengl_sphere ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${VTK_LIBRARIES} GL)
endif()

# qt examples
if(${BOOST_COMPUTE_HAVE_QT})
  find_package(Qt REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
  set(CMAKE_AUTOMOC TRUE)
  set(QT_USE_QTOPENGL TRUE)
  include(${QT_USE_FILE})
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  add_executable(qimage_blur qimage_blur.cpp)
  target_link_libraries(qimage_blur ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${QT_LIBRARIES})

  set(QT_OPENGL_EXAMPLES
    mandelbrot
    resize_image
  )
  foreach(EXAMPLE ${QT_OPENGL_EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${QT_LIBRARIES} GL)
  endforeach()
endif()
