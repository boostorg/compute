//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestCommandQueue
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(get_context)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);
    BOOST_VERIFY(queue.get_context() == context);
}

BOOST_AUTO_TEST_CASE(event_profiling)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device, bc::command_queue::enable_profiling);

    bc::event event = queue.enqueue_marker();
    queue.finish();

    event.get_profiling_info<cl_ulong>(bc::event::profiling_command_queued);
    event.get_profiling_info<cl_ulong>(bc::event::profiling_command_submit);
    event.get_profiling_info<cl_ulong>(bc::event::profiling_command_start);
    event.get_profiling_info<cl_ulong>(bc::event::profiling_command_end);
}
