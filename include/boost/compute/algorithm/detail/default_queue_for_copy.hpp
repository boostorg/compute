//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_DEFAULT_QUEUE_FOR_COPY_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_DEFAULT_QUEUE_FOR_COPY_HPP

#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class OutputIterator>
inline command_queue
default_queue_for_copy(InputIterator first,
                       OutputIterator result,
                       typename boost::enable_if<
                           typename boost::mpl::and_<
                               is_buffer_iterator<InputIterator>,
                               is_buffer_iterator<OutputIterator>
                           >
                       >::type* = 0)
{
    (void) first;

    const buffer &buffer = result.get_buffer();
    const context &context = buffer.get_context();
    const device &device = context.get_device();

    return command_queue(context, device);
}

template<class InputIterator, class OutputIterator>
inline command_queue
default_queue_for_copy(InputIterator first,
                       OutputIterator result,
                       typename boost::enable_if<
                           typename boost::mpl::and_<
                               boost::mpl::not_<is_buffer_iterator<InputIterator> >,
                               is_buffer_iterator<OutputIterator>
                           >
                       >::type* = 0)
{
    (void) first;

    const buffer &buffer = result.get_buffer();
    const context &context = buffer.get_context();
    const device &device = context.get_device();

    return command_queue(context, device);
}

template<class InputIterator, class OutputIterator>
inline command_queue
default_queue_for_copy(InputIterator first,
                       OutputIterator result,
                       typename boost::enable_if<
                           typename boost::mpl::and_<
                               is_buffer_iterator<InputIterator>,
                               boost::mpl::not_<is_buffer_iterator<OutputIterator> >
                           >
                       >::type* = 0)
{
    (void) result;

    const buffer &buffer = first.get_buffer();
    const context &context = buffer.get_context();
    const device &device = context.get_device();

    return command_queue(context, device);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_DEFAULT_QUEUE_FOR_COPY_HPP
