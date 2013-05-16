//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_FILL_HPP
#define BOOST_COMPUTE_ALGORITHM_FILL_HPP

#include <iterator>

#include <boost/utility/enable_if.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/iterator/constant_iterator.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

#if defined(CL_VERSION_1_2)
// specialization which uses clEnqueueFillBuffer for buffer iterators
template<class BufferIterator, class T>
void dispatch_fill(BufferIterator first,
                   size_t count,
                   const T &value,
                   command_queue &queue,
                   typename boost::enable_if_c<
                       is_buffer_iterator<BufferIterator>::value
                   >::type* = 0)
{
    typedef typename std::iterator_traits<BufferIterator>::value_type value_type;

    value_type pattern = static_cast<value_type>(value);
    size_t offset = static_cast<size_t>(first.get_index());

    queue.enqueue_fill_buffer(first.get_buffer(),
                              &pattern,
                              sizeof(value_type),
                              offset * sizeof(value_type),
                              count * sizeof(value_type));
}

// default implementation
template<class BufferIterator, class T>
void dispatch_fill(BufferIterator first,
                   size_t count,
                   const T &value,
                   command_queue &queue,
                   typename boost::disable_if_c<
                       is_buffer_iterator<BufferIterator>::value
                   >::type* = 0)
{
    ::boost::compute::copy(
        ::boost::compute::make_constant_iterator(value, 0),
        ::boost::compute::make_constant_iterator(value, count),
        first,
        queue
    );
}
#else
template<class BufferIterator, class T>
void dispatch_fill(BufferIterator first,
                   size_t count,
                   const T &value,
                   command_queue &queue)
{
    ::boost::compute::copy(
        ::boost::compute::make_constant_iterator(value, 0),
        ::boost::compute::make_constant_iterator(value, count),
        first,
        queue
    );
}
#endif // !defined(CL_VERSION_1_2)

} // end detail namespace

template<class BufferIterator, class T>
inline void fill(BufferIterator first,
                 BufferIterator last,
                 const T &value,
                 command_queue &queue = system::default_queue())
{
    size_t count = detail::iterator_range_size(first, last);

    detail::dispatch_fill(first, count, value, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_FILL_HPP
