//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_TRANSFORM_REDUCE_HPP
#define BOOST_COMPUTE_ALGORITHM_TRANSFORM_REDUCE_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/functional/detail/unpack.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {

/// Transforms each value in the range [\p first, \p last) with the unary
/// \p transform_function and then reduces each transformed value and
/// \p init with \p reduce_function.
template<class InputIterator,
         class OutputIterator,
         class UnaryTransformFunction,
         class T,
         class BinaryReduceFunction>
inline void transform_reduce(InputIterator first,
                             InputIterator last,
                             OutputIterator result,
                             UnaryTransformFunction transform_function,
                             T init,
                             BinaryReduceFunction reduce_function,
                             command_queue &queue = system::default_queue())
{
    ::boost::compute::reduce(
        ::boost::compute::make_transform_iterator(first, transform_function),
        ::boost::compute::make_transform_iterator(last, transform_function),
        result,
        init,
        reduce_function,
        queue
    );
}

/// Transforms each value in the range [\p first1, \p last1) and the
/// range beginning at \p first2 with the binary \p transform_function
/// and then reduces each transformed value and \p init with
/// \p reduce_function.
template<class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryTransformFunction,
         class T,
         class BinaryReduceFunction>
inline void transform_reduce(InputIterator1 first1,
                             InputIterator1 last1,
                             InputIterator2 first2,
                             OutputIterator result,
                             BinaryTransformFunction transform_function,
                             T init,
                             BinaryReduceFunction reduce_function,
                             command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator1>::difference_type difference_type;

    difference_type n = std::distance(first1, last1);

    ::boost::compute::transform_reduce(
        ::boost::compute::make_zip_iterator(
            boost::make_tuple(first1, first2)
        ),
        ::boost::compute::make_zip_iterator(
            boost::make_tuple(last1, first2 + n)
        ),
        result,
        detail::unpack(transform_function),
        init,
        reduce_function,
        queue
    );
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_TRANSFORM_REDUCE_HPP
