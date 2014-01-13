//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_SORT_HPP
#define BOOST_COMPUTE_ALGORITHM_SORT_HPP

#include <iterator>

#include <boost/utility/enable_if.hpp>

#include <boost/compute/buffer.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/detail/fixed_sort.hpp>
#include <boost/compute/algorithm/detail/radix_sort.hpp>
#include <boost/compute/algorithm/detail/insertion_sort.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

// sort() for device iterators
template <class Iterator>
inline void dispatch_sort(Iterator first,
                          Iterator last,
                          command_queue &queue,
                          typename boost::enable_if<
                              is_device_iterator<Iterator>
                          >::type* = 0)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    size_t count = detail::iterator_range_size(first, last);
    if(count < 2){
        return;
    }
    else if(count == 2){
        ::boost::compute::detail::sort2<T>(first.get_buffer(), queue);
    }
    else if(count == 3){
        ::boost::compute::detail::sort3<T>(first.get_buffer(), queue);
    }
    else if(count <= 32){
        ::boost::compute::detail::serial_insertion_sort(first, last, queue);
    }
    else {
        ::boost::compute::detail::radix_sort(first, last, queue);
    }
}

// sort() for host iterators
template <class Iterator>
inline void dispatch_sort(Iterator first,
                          Iterator last,
                          command_queue &queue,
                          typename boost::disable_if<
                              is_device_iterator<Iterator>
                          >::type* = 0)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    size_t size = static_cast<size_t>(std::distance(first, last));

    // create mapped buffer
    buffer mapped_buffer(
        queue.get_context(),
        size * sizeof(T),
        buffer::read_write | buffer::use_host_ptr,
        boost::addressof(*first)
    );

    // sort mapped buffer
    dispatch_sort(
        make_buffer_iterator<T>(mapped_buffer, 0),
        make_buffer_iterator<T>(mapped_buffer, size),
        queue
    );

    // return results to host
    queue.enqueue_map_buffer(
        mapped_buffer,
        command_queue::map_read | command_queue::map_write,
        0,
        size * sizeof(T)
    );
}

} // end detail namespace

/// Sorts the values in the range [\p first, \p last) according to
/// \p compare.
template<class Iterator, class Compare>
inline void sort(Iterator first,
                 Iterator last,
                 Compare compare,
                 command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    size_t count = detail::iterator_range_size(first, last);
    if(count < 2){
        return;
    }

    return ::boost::compute::detail::serial_insertion_sort(first,
                                                           last,
                                                           compare,
                                                           queue);
}

/// Sorts the values in the range [\p first, \p last).
template<class Iterator>
inline void sort(Iterator first,
                 Iterator last,
                 command_queue &queue = system::default_queue())
{
    detail::dispatch_sort(first, last, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_SORT_HPP
