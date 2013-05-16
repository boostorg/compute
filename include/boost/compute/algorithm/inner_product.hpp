//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_INNER_PRODUCT_HPP
#define BOOST_COMPUTE_ALGORITHM_INNER_PRODUCT_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform_reduce.hpp>
#include <boost/compute/container/vector.hpp>

namespace boost {
namespace compute {

template<class InputIterator1, class InputIterator2, class T>
inline T inner_product(InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       T init,
                       command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator1>::value_type input_type;
    typedef typename multiplies<input_type>::result_type product_type;

    return ::boost::compute::transform_reduce(first1,
                                              last1,
                                              first2,
                                              multiplies<input_type>(),
                                              init,
                                              plus<product_type>(),
                                              queue);
}

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
