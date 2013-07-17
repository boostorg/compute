//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_TRANSFORM_HPP
#define BOOST_COMPUTE_ALGORITHM_TRANSFORM_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>
#include <boost/compute/iterator/detail/binary_transform_iterator.hpp>

namespace boost {
namespace compute {

/// Transforms the elements in the range [\p first, \p last) using
/// \p transform and stores the results in the range beginning at
/// \p result.
template<class InputIterator, class OutputIterator, class UnaryOperator>
inline OutputIterator transform(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                UnaryOperator op,
                                command_queue &queue = system::default_queue())
{
    return copy(
               ::boost::compute::make_transform_iterator(first, op),
               ::boost::compute::make_transform_iterator(last, op),
               result,
               queue
           );
}

/// Transforms the elements in the range [\p first1, \p last1) and
/// \p first2 using \p op and stores the results in the range beginning
/// at \p result.
template<class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryOperator>
inline OutputIterator transform(InputIterator1 first1,
                                InputIterator1 last1,
                                InputIterator2 first2,
                                OutputIterator result,
                                BinaryOperator op,
                                command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator1>::difference_type difference_type;

    difference_type n = std::distance(first1, last1);

    return copy(
               detail::make_binary_transform_iterator(first1, first2, op),
               detail::make_binary_transform_iterator(last1, last1 + n, op),
               result,
               queue
           );
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_TRANSFORM_HPP
