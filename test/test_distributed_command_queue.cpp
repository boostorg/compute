//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestDistributedCommandQueue
#include <boost/test/unit_test.hpp>

#include <boost/compute/algorithm.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(construct_from_distributed_context)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);

    bc::distributed::context distributed_context(contexts);
    bc::distributed::command_queue distributed_queue1(distributed_context);
    BOOST_CHECK_EQUAL(
        distributed_queue1.size(),
        distributed_context.get_devices(0).size()
    );

    bc::distributed::command_queue distributed_queue2(
        distributed_context, bc::distributed::command_queue::enable_profiling
    );
    BOOST_CHECK_EQUAL(
        distributed_queue2.size(),
        distributed_context.get_devices(0).size()
    );
    BOOST_CHECK_EQUAL(
        distributed_queue2
            .get_info<cl_command_queue_properties>(0, CL_QUEUE_PROPERTIES),
        bc::distributed::command_queue::enable_profiling
    );
}

BOOST_AUTO_TEST_CASE(construct_from_contexts)
{
    std::vector<bc::context> contexts;
    std::vector<bc::device> devices;

    contexts.push_back(context);
    devices.push_back(device);

    bc::distributed::command_queue distributed_queue1(contexts, devices);
    BOOST_CHECK_EQUAL(
        distributed_queue1.size(),
        1
    );

    contexts.push_back(context);
    devices.push_back(device);

    bc::distributed::command_queue distributed_queue2(
        contexts, devices, bc::distributed::command_queue::enable_profiling
    );
    BOOST_CHECK_EQUAL(
        distributed_queue2.size(),
        2
    );
    BOOST_CHECK_EQUAL(
        distributed_queue2
            .get_info<cl_command_queue_properties>(0, CL_QUEUE_PROPERTIES),
        bc::distributed::command_queue::enable_profiling
    );
}

BOOST_AUTO_TEST_CASE(construct_from_command_queues)
{
    std::vector<bc::command_queue> queues;
    queues.push_back(queue);

    bc::distributed::command_queue distributed_queue1(queues);
    BOOST_CHECK_EQUAL(distributed_queue1.size(), 1);

    queues.push_back(queue);
    bc::distributed::command_queue distributed_queue2(queues);
    BOOST_CHECK_EQUAL(distributed_queue2.size(), 2);
}

BOOST_AUTO_TEST_CASE(get_context)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);
    contexts.push_back(context);

    bc::distributed::context distributed_context(contexts);
    bc::distributed::command_queue distributed_queue(distributed_context);

    BOOST_CHECK(
        distributed_queue.get_context() == distributed_context
    );
    BOOST_CHECK(
        distributed_queue.get_context(0) == context
    );
    BOOST_CHECK(
        distributed_queue.get_context(1) == context
    );
}

BOOST_AUTO_TEST_CASE(get_devices)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);
    contexts.push_back(context);

    bc::distributed::context distributed_context(contexts);
    bc::distributed::command_queue distributed_queue(distributed_context);

    BOOST_CHECK(
        distributed_queue.get_device(0) == context.get_device()
    );
    BOOST_CHECK(
        distributed_queue.get_device(1) == context.get_device()
    );
}

BOOST_AUTO_TEST_CASE(get_command_queue)
{
    std::vector<bc::command_queue> queues;
    queues.push_back(queue);
    queues.push_back(queue);

    bc::distributed::command_queue distributed_queue(queues);

    BOOST_CHECK(
        distributed_queue.get(0) == queue
    );
    BOOST_CHECK(
        distributed_queue.get(1) == queue
    );
}

BOOST_AUTO_TEST_CASE(enqueue_kernel)
{
    std::vector<bc::command_queue> queues;
    queues.push_back(queue);
    queues.push_back(queue);

    bc::distributed::command_queue distributed_queue(queues);

    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void foo(__global int *output)
        {
            output[get_global_id(0)] = get_global_id(0);
        }
    );

    for(size_t i = 0; i < distributed_queue.size(); i++)
    {
        bc::command_queue& queue = distributed_queue.get(0);
        bc::vector<bc::uint_> output(8, queue.get_context());

        bc::kernel kernel =
            bc::kernel::create_with_source(source, "foo", queue.get_context());

        kernel.set_arg(0, output);
        queue.enqueue_1d_range_kernel(kernel, 0, output.size(), 0);
        CHECK_RANGE_EQUAL(bc::uint_, 8, output, (0, 1, 2, 3, 4, 5, 6, 7));
    }
}

BOOST_AUTO_TEST_SUITE_END()
