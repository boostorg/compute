include_directories(../include)

set(BOOST_COMPONENTS unit_test_framework)

if (${BOOST_COMPUTE_USE_OFFLINE_CACHE})
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} system filesystem)
    add_definitions(-DBOOST_COMPUTE_USE_OFFLINE_CACHE)
endif()

find_package(Boost 1.48 REQUIRED COMPONENTS ${BOOST_COMPONENTS})
add_definitions(-DBOOST_TEST_DYN_LINK)

add_definitions(-DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION)

# enable code coverage generation (only with GCC for now)
if(${BOOST_COMPUTE_ENABLE_COVERAGE} AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_definitions(-fprofile-arcs -ftest-coverage)
endif()

function(add_compute_test TEST_NAME TEST_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET}
    ${OPENCL_LIBRARIES}
    ${Boost_LIBRARIES}
  )

  # link with coverage library
  if(${BOOST_COMPUTE_ENABLE_COVERAGE} AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_link_libraries(${TEST_TARGET} -fprofile-arcs -ftest-coverage)
  endif()

  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

add_compute_test("core.buffer" test_buffer.cpp)
add_compute_test("core.closure" test_closure.cpp)
add_compute_test("core.command_queue" test_command_queue.cpp)
add_compute_test("core.context" test_context.cpp)
add_compute_test("core.device" test_device.cpp)
add_compute_test("core.event" test_event.cpp)
add_compute_test("core.function" test_function.cpp)
add_compute_test("core.image2d" test_image2d.cpp)
add_compute_test("core.image3d" test_image3d.cpp)
add_compute_test("core.image_sampler" test_image_sampler.cpp)
add_compute_test("core.kernel" test_kernel.cpp)
add_compute_test("core.program" test_program.cpp)
add_compute_test("core.system" test_system.cpp)
add_compute_test("core.type_traits" test_type_traits.cpp)
add_compute_test("core.user_event" test_user_event.cpp)
add_compute_test("core.wait_list" test_wait_list.cpp)
add_compute_test("core.context_error" test_context_error.cpp)
add_compute_test("core.unsupported_extension" test_unsupported_extension.cpp)

add_compute_test("algorithm.accumulate" test_accumulate.cpp)
add_compute_test("algorithm.adjacent_difference" test_adjacent_difference.cpp)
add_compute_test("algorithm.adjacent_find" test_adjacent_find.cpp)
add_compute_test("algorithm.any_all_none_of" test_any_all_none_of.cpp)
add_compute_test("algorithm.binary_search" test_binary_search.cpp)
add_compute_test("algorithm.copy" test_copy.cpp)
add_compute_test("algorithm.copy_if" test_copy_if.cpp)
add_compute_test("algorithm.count" test_count.cpp)
add_compute_test("algorithm.equal" test_equal.cpp)
add_compute_test("algorithm.equal_range" test_equal_range.cpp)
add_compute_test("algorithm.extrema" test_extrema.cpp)
add_compute_test("algorithm.fill" test_fill.cpp)
add_compute_test("algorithm.find" test_find.cpp)
add_compute_test("algorithm.find_end" test_find_end.cpp)
add_compute_test("algorithm.for_each" test_for_each.cpp)
add_compute_test("algorithm.gather" test_gather.cpp)
add_compute_test("algorithm.generate" test_generate.cpp)
add_compute_test("algorithm.includes" test_includes.cpp)
add_compute_test("algorithm.inner_product" test_inner_product.cpp)
add_compute_test("algorithm.inplace_merge" test_inplace_merge.cpp)
add_compute_test("algorithm.inplace_reduce" test_inplace_reduce.cpp)
add_compute_test("algorithm.insertion_sort" test_insertion_sort.cpp)
add_compute_test("algorithm.iota" test_iota.cpp)
add_compute_test("algorithm.is_permutation" test_is_permutation.cpp)
add_compute_test("algorithm.is_sorted" test_is_sorted.cpp)
add_compute_test("algorithm.merge" test_merge.cpp)
add_compute_test("algorithm.mismatch" test_mismatch.cpp)
add_compute_test("algorithm.next_permutation" test_next_permutation.cpp)
add_compute_test("algorithm.nth_element" test_nth_element.cpp)
add_compute_test("algorithm.partial_sum" test_partial_sum.cpp)
add_compute_test("algorithm.partition" test_partition.cpp)
add_compute_test("algorithm.partition_point" test_partition_point.cpp)
add_compute_test("algorithm.prev_permutation" test_prev_permutation.cpp)
add_compute_test("algorithm.radix_sort" test_radix_sort.cpp)
add_compute_test("algorithm.random_fill" test_random_fill.cpp)
add_compute_test("algorithm.random_shuffle" test_random_shuffle.cpp)
add_compute_test("algorithm.reduce" test_reduce.cpp)
add_compute_test("algorithm.remove" test_remove.cpp)
add_compute_test("algorithm.replace" test_replace.cpp)
add_compute_test("algorithm.reverse" test_reverse.cpp)
add_compute_test("algorithm.rotate" test_rotate.cpp)
add_compute_test("algorithm.rotate_copy" test_rotate_copy.cpp)
add_compute_test("algorithm.scan" test_scan.cpp)
add_compute_test("algorithm.scatter" test_scatter.cpp)
add_compute_test("algorithm.search" test_search.cpp)
add_compute_test("algorithm.search_n" test_search_n.cpp)
add_compute_test("algorithm.set_difference" test_set_difference.cpp)
add_compute_test("algorithm.set_intersection" test_set_intersection.cpp)
add_compute_test("algorithm.set_symmetric_difference" test_set_symmetric_difference.cpp)
add_compute_test("algorithm.set_union" test_set_union.cpp)
add_compute_test("algorithm.sort" test_sort.cpp)
add_compute_test("algorithm.sort_by_key" test_sort_by_key.cpp)
add_compute_test("algorithm.stable_partition" test_stable_partition.cpp)
add_compute_test("algorithm.stable_sort" test_stable_sort.cpp)
add_compute_test("algorithm.transform" test_transform.cpp)
add_compute_test("algorithm.transform_reduce" test_transform_reduce.cpp)
add_compute_test("algorithm.unique" test_unique.cpp)
add_compute_test("algorithm.unique_copy" test_unique_copy.cpp)

add_compute_test("async.wait" test_async_wait.cpp)

add_compute_test("container.allocator" test_allocator.cpp)
add_compute_test("container.array" test_array.cpp)
add_compute_test("container.flat_map" test_flat_map.cpp)
add_compute_test("container.flat_set" test_flat_set.cpp)
add_compute_test("container.mapped_view" test_mapped_view.cpp)
add_compute_test("container.stack" test_stack.cpp)
add_compute_test("container.string" test_string.cpp)
add_compute_test("container.valarray" test_valarray.cpp)
add_compute_test("container.vector" test_vector.cpp)

add_compute_test("exception.opencl_error" test_opencl_error.cpp)

add_compute_test("functional.as" test_functional_as.cpp)
add_compute_test("functional.bind" test_functional_bind.cpp)
add_compute_test("functional.convert" test_functional_convert.cpp)
add_compute_test("functional.get" test_functional_get.cpp)
add_compute_test("functional.hash" test_functional_hash.cpp)
add_compute_test("functional.unpack" test_functional_unpack.cpp)

add_compute_test("iterator.adjacent_transform_iterator" test_adjacent_transform_iterator.cpp)
add_compute_test("iterator.buffer_iterator" test_buffer_iterator.cpp)
add_compute_test("iterator.constant_iterator" test_constant_iterator.cpp)
add_compute_test("iterator.counting_iterator" test_counting_iterator.cpp)
add_compute_test("iterator.discard_iterator" test_discard_iterator.cpp)
add_compute_test("iterator.permutation_iterator" test_permutation_iterator.cpp)
add_compute_test("iterator.transform_iterator" test_transform_iterator.cpp)
add_compute_test("iterator.zip_iterator" test_zip_iterator.cpp)

add_compute_test("random.bernoulli_distribution" test_bernoulli_distribution.cpp)
add_compute_test("random.discrete_distribution" test_discrete_distribution.cpp)
add_compute_test("random.linear_congruential_engine" test_linear_congruential_engine.cpp)
add_compute_test("random.mersenne_twister_engine" test_mersenne_twister_engine.cpp)
add_compute_test("random.normal_distribution" test_normal_distribution.cpp)
add_compute_test("random.uniform_int_distribution" test_uniform_int_distribution.cpp)
add_compute_test("random.uniform_real_distribution" test_uniform_real_distribution.cpp)

add_compute_test("types.builtin" test_types.cpp)
add_compute_test("types.complex" test_complex.cpp)
add_compute_test("types.pair" test_pair.cpp)
add_compute_test("types.tuple" test_tuple.cpp)
add_compute_test("types.struct" test_struct.cpp)

add_compute_test("experimental.clamp_range" test_clamp_range.cpp)
add_compute_test("experimental.malloc" test_malloc.cpp)
add_compute_test("experimental.sort_by_transform" test_sort_by_transform.cpp)
add_compute_test("experimental.tabulate" test_tabulate.cpp)
add_compute_test("experimental.transform_if" test_transform_if.cpp)

add_compute_test("ext.lambda" test_lambda.cpp)
add_compute_test("ext.program_cache" test_program_cache.cpp)
add_compute_test("ext.amd_cpp_kernel_language" test_amd_cpp_kernel_language.cpp)

# eigen interop tests
if(${BOOST_COMPUTE_HAVE_EIGEN})
  find_package(Eigen REQUIRED)
  include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
  add_compute_test("interop.eigen" test_interop_eigen.cpp)
endif()

# opencv interop tests
if(${BOOST_COMPUTE_HAVE_OPENCV})
  find_package(OpenCV REQUIRED)
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  add_compute_test("interop.opencv" test_interop_opencv.cpp)
  target_link_libraries(test_interop_opencv ${OpenCV_LIBS})
endif()

# qt interop tests
if(${BOOST_COMPUTE_HAVE_QT})
  find_package(Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
  include(${QT_USE_FILE})
  add_compute_test("interop.qt" test_interop_qt.cpp)
  target_link_libraries(test_interop_qt ${QT_LIBRARIES})

  # the opengl interop test depends on qt to create the opengl context
  add_compute_test("interop.opengl" test_interop_opengl.cpp)
  target_link_libraries(test_interop_opengl ${QT_LIBRARIES} GL)
endif()

# vtk interop tests
if(${BOOST_COMPUTE_HAVE_VTK})
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
  add_compute_test("interop.vtk" test_interop_vtk.cpp)
  target_link_libraries(test_interop_vtk ${VTK_LIBRARIES})
endif()

# tests for particular bugs/issues
add_compute_test("issue.user_defined_types" test_user_defined_types.cpp)

# Check for linkage problems
add_executable(test_multiple_objects
    test_multiple_objects1.cpp
    test_multiple_objects2.cpp
    )
target_link_libraries(test_multiple_objects
    ${OPENCL_LIBRARIES}
    ${Boost_LIBRARIES}
    )
# link with coverage library
if(${BOOST_COMPUTE_ENABLE_COVERAGE} AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_link_libraries(test_multiple_objects -fprofile-arcs -ftest-coverage)
endif()
add_test("issue.multiple_objects" test_multiple_objects)
