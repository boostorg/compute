//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestDistributedVector
#include <boost/test/unit_test.hpp>

#include <boost/compute/algorithm.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>
#include <boost/compute/container/vector.hpp>

#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/vector.hpp>
#include <boost/compute/distributed/copy.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

#include "distributed_check_functions.hpp"
#include "distributed_queue_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(copy_from_host)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value = -1;
    bc::distributed::vector<bc::int_> distributed_vector(
        size, value, distributed_queue
    );
    distributed_queue.finish();

    std::vector<bc::int_> host(size_t(size), bc::int_(1000));
    host[size - 1] = -10;
    host[0] = -20;
    host[size/2] = -30;

    // empty copy
    bc::distributed::copy(
        host.begin(), host.begin(), distributed_vector, distributed_queue
    );
    BOOST_CHECK(distributed_equal(distributed_vector, value, distributed_queue));

    // full copy
    bc::distributed::copy(
        host.begin(), host.end(), distributed_vector, distributed_queue
    );
    BOOST_CHECK(
        distributed_equal(
            distributed_vector,
            host.begin(), host.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_async_from_host)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value = -1;
    bc::distributed::vector<bc::int_> distributed_vector(
        size, value, distributed_queue
    );
    distributed_queue.finish();

    std::vector<bc::int_> host(size_t(size), bc::int_(1000));
    host[size - 1] = -10;
    host[0] = -20;
    host[size/2] = -30;

    // empty copy
    std::vector<boost::compute::event> events =
        bc::distributed::copy_async(
            host.begin(), host.begin(), distributed_vector, distributed_queue
        );
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
    BOOST_CHECK(distributed_equal(distributed_vector, value, distributed_queue));

    // full copy
    events =
        bc::distributed::copy_async(
            host.begin(), host.end(), distributed_vector, distributed_queue
        );
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
    BOOST_CHECK(
        distributed_equal(
            distributed_vector,
            host.begin(), host.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_to_host)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value = -1;
    bc::distributed::vector<bc::int_> distributed_vector(
        size, value, distributed_queue
    );
    distributed_queue.finish();

    std::vector<bc::int_> host(size, bc::int_(1000));
    bc::distributed::copy(
        distributed_vector, host.begin(), distributed_queue
    );
    for(size_t i = 0; i < host.size(); i++) {
        BOOST_CHECK_EQUAL(host[i], value);
    }
}

BOOST_AUTO_TEST_CASE(copy_async_to_host)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value = -1;
    bc::distributed::vector<bc::int_> distributed_vector(
        size, value, distributed_queue
    );
    distributed_queue.finish();

    std::vector<bc::int_> host(size, bc::int_(1000));
    std::vector<boost::compute::event> events =
        bc::distributed::copy_async(
            distributed_vector, host.begin(), distributed_queue
        );
    // wait for copy
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
    // check values
    for(size_t i = 0; i < host.size(); i++) {
        BOOST_CHECK_EQUAL(host[i], value);
    }
}

BOOST_AUTO_TEST_CASE(copy_from_vector_to_vector)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value1 = -1;
    bc::int_ value2 = 1;
    bc::distributed::vector<bc::int_> distributed_vector1(
        size, value1, distributed_queue
    );
    bc::distributed::vector<bc::int_> distributed_vector2(
        size, value2, distributed_queue
    );
    distributed_queue.finish();

    BOOST_CHECK(distributed_equal(distributed_vector1, value1, distributed_queue));
    BOOST_CHECK(distributed_equal(distributed_vector2, value2, distributed_queue));

    bc::distributed::copy(
        distributed_vector1, distributed_vector2, distributed_queue
    );

    // check if distributed_vector1 is the same
    BOOST_CHECK(distributed_equal(distributed_vector1, value1, distributed_queue));
    // and it was copied into distributed_vector2
    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            distributed_vector2,
            distributed_queue
        )
    );

    // change distributed_vector1
    distributed_vector1
        .begin(distributed_vector1.parts() - 1)
        .write(99, distributed_queue.get(distributed_vector1.parts() - 1));
    distributed_queue.get(distributed_vector1.parts() - 1).finish();

    // copy once again
    bc::distributed::copy(
        distributed_vector1, distributed_vector2, distributed_queue
    );

    // check
    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            distributed_vector2,
            distributed_queue
        )
    );

    // copy between vectors of different sizes
    bc::int_ value3 = 12;
    bc::distributed::vector<bc::int_> distributed_vector3(
        2 * size, value3, distributed_queue
    );

    bc::distributed::copy(
        distributed_vector1, distributed_vector3, distributed_queue
    );

    // check
    std::vector<bc::int_> host(distributed_vector3.size(), value3);
    bc::distributed::copy(
        distributed_vector1, host.begin(), distributed_queue
    );
    BOOST_CHECK(
        distributed_equal(
            distributed_vector3,
            host.begin(), host.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_async_from_vector_to_vector)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value1 = -1;
    bc::int_ value2 = 1;
    bc::distributed::vector<bc::int_> distributed_vector1(
        size, value1, distributed_queue
    );
    bc::distributed::vector<bc::int_> distributed_vector2(
        size, value2, distributed_queue
    );
    distributed_queue.finish();

    BOOST_CHECK(distributed_equal(distributed_vector1, value1, distributed_queue));
    BOOST_CHECK(distributed_equal(distributed_vector2, value2, distributed_queue));

    std::vector<boost::compute::event> events = bc::distributed::copy_async(
        distributed_vector1, distributed_vector2, distributed_queue
    );
    // wait for copy
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }

    // check if distributed_vector1 is the same
    BOOST_CHECK(distributed_equal(distributed_vector1, value1, distributed_queue));
    // and it was copied into distributed_vector2
    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            distributed_vector2,
            distributed_queue
        )
    );

    // change distributed_vector1
    distributed_vector1
        .begin(distributed_vector1.parts() - 1)
        .write(99, distributed_queue.get(distributed_vector1.parts() - 1));
    distributed_queue.get(distributed_vector1.parts() - 1).finish();

    // copy once again
    events = bc::distributed::copy_async(
        distributed_vector1, distributed_vector2, distributed_queue
    );
    // wait for copy
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }

    // check
    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            distributed_vector2,
            distributed_queue
        )
    );
}

std::vector<double> custom_weight_func(const bc::distributed::command_queue& queue)
{
    return std::vector<double>(queue.size(), 1.0/queue.size());
}

BOOST_AUTO_TEST_CASE(copy_from_vector_to_vector_different_vectors)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value1 = -1;
    bc::long_ value2 = 1;
    bc::distributed::vector<bc::int_> distributed_vector1(
        size, value1, distributed_queue
    );
    bc::distributed::vector<bc::long_, custom_weight_func> distributed_vector2(
        size, value2, distributed_queue
    );
    distributed_queue.finish();

    BOOST_CHECK(distributed_equal(distributed_vector1, value1, distributed_queue));
    BOOST_CHECK(distributed_equal(distributed_vector2, value2, distributed_queue));

    bc::distributed::copy(
        distributed_vector1, distributed_vector2, distributed_queue, distributed_queue
    );

    // check if distributed_vector1 is the same
    BOOST_CHECK(distributed_equal(distributed_vector1, value1, distributed_queue));
    // and it was copied into distributed_vector2
    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            distributed_vector2,
            distributed_queue,
            distributed_queue
        )
    );

    // change distributed_vector1
    distributed_vector1
        .begin(distributed_vector1.parts() - 1)
        .write(99, distributed_queue.get(distributed_vector1.parts() - 1));
    distributed_queue.get(distributed_vector1.parts() - 1).finish();

    // copy once again
    bc::distributed::copy(
        distributed_vector1, distributed_vector2, distributed_queue, distributed_queue
    );

    // check
    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            distributed_vector2,
            distributed_queue,
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_device_to_vector)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value1 = -1;
    bc::long_ value2 = 1;
    bc::vector<bc::int_> device_vector(size, value1, queue);
    bc::distributed::vector<bc::long_> distributed_vector(
        size, value2, distributed_queue
    );

    bc::distributed::copy(
        device_vector.begin(), device_vector.end(), distributed_vector,
        queue, distributed_queue
    );

    BOOST_CHECK(
        distributed_equal(
            distributed_vector, bc::long_(value1), distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_async_device_to_vector)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue, 2, true);

    size_t size = 64;
    bc::int_ value1 = -1;
    bc::long_ value2 = 1;
    bc::vector<bc::int_> device_vector(size, value1, queue);
    bc::distributed::vector<bc::long_> distributed_vector(
        size, value2, distributed_queue
    );

    std::vector<bc::event> events =
        bc::distributed::copy_async(
            device_vector.begin(), device_vector.end(), distributed_vector,
            queue, distributed_queue
        );
    // wait for copy
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }

    BOOST_CHECK(
        distributed_equal(
            distributed_vector, bc::long_(value1), distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_vector_to_device)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    size_t size = 64;
    bc::int_ value1 = -1;
    bc::long_ value2 = 1;
    bc::vector<bc::int_> device_vector(size, value1, queue);
    bc::distributed::vector<bc::long_> distributed_vector(
        size, value2, distributed_queue
    );

    BOOST_CHECK(
        distributed_equal(
            distributed_vector, value2, distributed_queue
        )
    );
    BOOST_CHECK(
        boost::compute::equal(
            device_vector.begin(),
            device_vector.end(),
            boost::compute::make_constant_iterator(value1),
            queue
        )
    );

    bc::distributed::copy(
        distributed_vector, device_vector.begin(),
        distributed_queue, queue
    );

    BOOST_CHECK(
        boost::compute::equal(
            device_vector.begin(),
            device_vector.end(),
            boost::compute::make_constant_iterator(static_cast<bc::int_>(value2)),
            queue
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_async_vector_to_device)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue, 2, true);

    size_t size = 64;
    bc::int_ value1 = -1;
    bc::long_ value2 = 1;
    bc::vector<bc::int_> device_vector(size, value1, queue);
    bc::distributed::vector<bc::long_> distributed_vector(
        size, value2, distributed_queue
    );

    BOOST_CHECK(
        distributed_equal(
            distributed_vector, value2, distributed_queue
        )
    );
    BOOST_CHECK(
        boost::compute::equal(
            device_vector.begin(),
            device_vector.end(),
            boost::compute::make_constant_iterator(value1),
            queue
        )
    );

    std::vector<bc::event> events =
        bc::distributed::copy_async(
            distributed_vector, device_vector.begin(),
            distributed_queue, queue
        );
    // wait for copy
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }

    BOOST_CHECK(
        boost::compute::equal(
            device_vector.begin(),
            device_vector.end(),
            boost::compute::make_constant_iterator(static_cast<bc::int_>(value2)),
            queue
        )
    );
}

BOOST_AUTO_TEST_SUITE_END()
