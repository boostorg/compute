//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_DEFAULT_QUEUE_FOR_ITERATOR_HPP
#define BOOST_COMPUTE_DETAIL_DEFAULT_QUEUE_FOR_ITERATOR_HPP

#include <boost/utility/enable_if.hpp>

#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class Iterator>
inline command_queue
default_queue_for_iterator(const Iterator &iterator,
                           typename boost::enable_if<
                               detail::is_buffer_iterator<Iterator>
                           >::type* = 0)
{
    const buffer &buffer = iterator.get_buffer();
    const context &context = buffer.get_context();
    const device &device = context.get_device();

    return command_queue(context, device);
}

template<class Iterator>
inline command_queue
default_queue_for_iterator(const Iterator &iterator,
                           typename boost::disable_if<
                               detail::is_buffer_iterator<Iterator>
                           >::type* = 0)
{
    (void) iterator;

    return command_queue(system::default_context(), system::default_device());
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_DEFAULT_QUEUE_FOR_ITERATOR_HPP
