//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_STABLE_SORT_HPP
#define BOOST_COMPUTE_ALGORITHM_STABLE_SORT_HPP

#include <iterator>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/detail/insertion_sort.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class Iterator, class Compare>
inline void stable_sort(Iterator first,
                        Iterator last,
                        Compare compare,
                        command_queue &queue)
{
    return ::boost::compute::detail::serial_insertion_sort(first,
                                                           last,
                                                           compare,
                                                           queue);
}

template<class Iterator, class Compare>
inline void stable_sort(Iterator first,
                        Iterator last,
                        Compare compare)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    ::boost::compute::stable_sort(first, last, compare, queue);
}

template<class Iterator>
inline void stable_sort(Iterator first,
                        Iterator last,
                        command_queue &queue)
{
    typedef typename std::iterator_traits<Iterator>::value_type value_type;

    ::boost::compute::less<value_type> less;

    return ::boost::compute::stable_sort(first, last, less, queue);
}

template<class Iterator>
inline void stable_sort(Iterator first,
                        Iterator last)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    ::boost::compute::stable_sort(first, last, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_STABLE_SORT_HPP
