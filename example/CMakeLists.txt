include_directories(../include)

set(EXAMPLES
  black_scholes
  copy_data
  hello_world
  host_sort
  inline_ptx
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

# eigen examples
if(${BOOST_COMPUTE_HAVE_EIGEN})
  find_package(Eigen REQUIRED)
  include_directories(${EIGEN_INCLUDE_DIRS})
  add_executable(batched_determinant batched_determinant.cpp)
  target_link_libraries(batched_determinant ${OPENCL_LIBRARIES})
endif()

# opencv examples
if(${BOOST_COMPUTE_HAVE_OPENCV})
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_executable(opencv_filter opencv_filter.cpp)
  target_link_libraries(opencv_filter ${OPENCL_LIBRARIES} ${OpenCV_LIBRARIES})
endif()

# opengl/vtk examples
if(${BOOST_COMPUTE_HAVE_VTK})
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
  add_executable(opengl_sphere opengl_sphere.cpp)
  target_link_libraries(opengl_sphere ${OPENCL_LIBRARIES} ${VTK_LIBRARIES} GL)
endif()

# qt examples
if(${BOOST_COMPUTE_HAVE_QT})
  find_package(Qt REQUIRED COMPONENTS QtCore QtGui)
  include(${QT_USE_FILE})
  add_executable(qimage_blur qimage_blur.cpp)
  target_link_libraries(qimage_blur ${OPENCL_LIBRARIES} ${QT_LIBRARIES})
endif()
