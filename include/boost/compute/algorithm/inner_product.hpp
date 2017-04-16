//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_INNER_PRODUCT_HPP
#define BOOST_COMPUTE_ALGORITHM_INNER_PRODUCT_HPP

#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/system.hpp>

namespace boost {
namespace compute {

/// Returns the inner product of the elements in the range
/// [\p first1, \p last1) with the elements in the range beginning
/// at \p first2.
///
/// Space complexity: \Omega(1)<br>
/// Space complexity when binary operator is recognized as associative: \Omega(n)
template<class InputIterator1, class InputIterator2, class T>
inline T inner_product(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       T init,
                       command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator1>::value_type input_type;

    size_t count = detail::iterator_range_size(first1, last1);

    const context &context = queue.get_context();

    buffer buffer_a(context, count * sizeof(input_type));
    buffer buffer_b(context, count * sizeof(input_type));

    buffer buffer_c(context, count * sizeof(product_type));

    buffer_iterator<input_type> start_a = make_buffer_iterator<T>(buffer_a, 0);
    buffer_iterator<input_type> start_b = make_buffer_iterator<T>(buffer_b, 0);
    buffer_iterator<input_type> start_c = make_buffer_iterator<T>(buffer_c, 0);

    std::string source = 
        std::string("__kernel void inner_product(__global ") + 
            type_name<input_type>() + " *a, __global " + 
            type_name<input_type>() + " *b, __global " + 
            type_name<product_type>() + " *c)\n"
        "{\n"
        "   const uint i = get_global_id(0);\n"
        "   c[i] = a[i] * b[i];\n"
        "}\n";
    std::cout<<source<<std::endl;

    program program = program::create_with_source(source.c_str(), context);
    program.build();

    kernel kernel(program, "inner_product");
    kernel.set_arg(0, buffer_a);
    kernel.set_arg(1, buffer_b);
    kernel.set_arg(2, buffer_c);

    copy(first1, last1, start_a, queue);
    copy(first2, first2 + count, start_b, queue);

    queue.enqueue_1d_range_kernel(kernel, 0, count);

    return accumulate(start_c, start_c + count, init, queue);
}

/// \overload
template<class InputIterator1,
         class InputIterator2,
         class T,
         class BinaryAccumulateFunction,
         class BinaryTransformFunction>
inline T inner_product(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       T init,
                       BinaryAccumulateFunction accumulate_function,
                       BinaryTransformFunction transform_function,
                       command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator1>::value_type value_type;

    size_t count = detail::iterator_range_size(first1, last1);
    vector<value_type> result(count, queue.get_context());
    transform(first1,
              last1,
              first2,
              result.begin(),
              transform_function,
              queue);

    return ::boost::compute::accumulate(result.begin(),
                                        result.end(),
                                        init,
                                        accumulate_function,
                                        queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_INNER_PRODUCT_HPP
