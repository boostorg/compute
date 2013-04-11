//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_ADJACENT_DIFFERENCE_HPP
#define BOOST_COMPUTE_ALGORITHM_ADJACENT_DIFFERENCE_HPP

#include <iterator>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/iterator/detail/adjacent_transform_iterator.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class InputIterator, class OutputIterator>
inline OutputIterator adjacent_difference(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result,
                                          command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    ::boost::compute::minus<value_type> op;

    return ::boost::compute::copy(
               detail::make_adjacent_transform_iterator(first, op),
               detail::make_adjacent_transform_iterator(last, op),
               result,
               queue
           );
}

template<class InputIterator, class OutputIterator>
inline OutputIterator adjacent_difference(InputIterator first,
                                          InputIterator last,
                                          OutputIterator result)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::adjacent_difference(first, last, result, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_ADJACENT_DIFFERENCE_HPP
