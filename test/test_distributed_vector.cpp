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

#include "check_macros.hpp"
#include "context_setup.hpp"

#include "distributed_check_functions.hpp"
#include "distributed_queue_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(empty_vector)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::distributed::vector<bc::uint_> distributed_vector(distributed_queue);

    BOOST_CHECK(distributed_vector.empty());
    BOOST_CHECK(distributed_vector.size() == 0);
    BOOST_CHECK(distributed_vector.parts() == 2);

    for(size_t i = 0; i < distributed_vector.parts(); i++)
    {
        BOOST_CHECK(distributed_vector.begin(i) == distributed_vector.end(i));
        BOOST_CHECK(distributed_vector.part_start(i) == 0);
        BOOST_CHECK(distributed_vector.part_size(i) == 0);
    }
}

BOOST_AUTO_TEST_CASE(count_ctor)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::distributed::vector<bc::uint_> distributed_vector(
        64, distributed_queue
    );

    BOOST_CHECK(!distributed_vector.empty());
    BOOST_CHECK(distributed_vector.size() == 64);
    BOOST_CHECK(distributed_vector.parts() == distributed_vector.get_queue().size());

    size_t size_sum = 0;
    for(size_t i = 0; i < distributed_vector.parts(); i++)
    {
        size_sum += distributed_vector.part_size(i);
    }
    BOOST_CHECK_EQUAL(distributed_vector.size(), size_sum);

    std::vector<bc::buffer> buffers = distributed_vector.get_buffers();
    for(size_t i = 0; i < distributed_vector.parts(); i++)
    {
        BOOST_CHECK(buffers[i].get() != 0);
        BOOST_CHECK(buffers[i] == distributed_vector.get_buffer(i));
    }
}

BOOST_AUTO_TEST_CASE(command_queue_ctor)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::uint_ value = 991;
    bc::distributed::vector<bc::uint_> distributed_vector(
        size_t(35), value, distributed_queue
    );

    BOOST_CHECK(!distributed_vector.empty());
    BOOST_CHECK(distributed_vector.size() == 35);
    BOOST_CHECK(distributed_vector.parts() == distributed_vector.get_queue().size());

    size_t size_sum = 0;
    for(size_t i = 0; i < distributed_vector.parts(); i++)
    {
        size_sum += distributed_vector.part_size(i);
    }
    BOOST_CHECK_EQUAL(distributed_vector.size(), size_sum);

    // need to finish since back() and front()
    // use different (self-made) queues
    distributed_queue.finish();
    BOOST_CHECK_EQUAL(distributed_vector.back(), value);
    BOOST_CHECK_EQUAL(distributed_vector.front(), value);

    BOOST_CHECK(distributed_equal(distributed_vector, value, distributed_queue));
}

BOOST_AUTO_TEST_CASE(command_queue_ctor_one_queue)
{
    // construct distributed::command_queue
    // with only 1 device command queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue, 0);

    bc::uint_ value = 1;
    bc::distributed::vector<bc::uint_> distributed_vector(
        size_t(5), value, distributed_queue
    );

    BOOST_CHECK(!distributed_vector.empty());
    BOOST_CHECK(distributed_vector.size() == 5);
    BOOST_CHECK(distributed_vector.parts() == 1);
    BOOST_CHECK_EQUAL(distributed_vector.size(), distributed_vector.part_size(0));

    // need to finish since back() and front()
    // use different (self-made) queues
    distributed_queue.finish();
    BOOST_CHECK_EQUAL(distributed_vector.back(), value);
    BOOST_CHECK_EQUAL(distributed_vector.front(), value);

    BOOST_CHECK(distributed_equal(distributed_vector, value, distributed_queue));
}

BOOST_AUTO_TEST_CASE(host_iterator_ctor)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::int_ value = -1;
    std::vector<bc::int_> host_vector(50, value);

    bc::distributed::vector<bc::int_> distributed_vector(
        host_vector.begin(), host_vector.end(), distributed_queue
    );

    BOOST_CHECK(!distributed_vector.empty());
    BOOST_CHECK(distributed_vector.size() == host_vector.size());
    BOOST_CHECK(distributed_vector.parts() == distributed_vector.get_queue().size());

    size_t size_sum = 0;
    for(size_t i = 0; i < distributed_vector.parts(); i++)
    {
        size_sum += distributed_vector.part_size(i);
    }
    BOOST_CHECK_EQUAL(distributed_vector.size(), size_sum);

    BOOST_CHECK(distributed_equal(distributed_vector, value, distributed_queue));

    // need to finish since back() and front()
    // use different (self-made) queues
    distributed_queue.finish();

    distributed_vector.front() = 1;
    distributed_vector.back() = 1;
    BOOST_CHECK_EQUAL(distributed_vector.back(), 1);
    BOOST_CHECK_EQUAL(distributed_vector.front(), 1);
}

BOOST_AUTO_TEST_CASE(copy_ctor)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue1 =
        get_distributed_queue(queue);
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue2 =
        get_distributed_queue(queue, 2);

    bc::int_ value = -1;
    size_t size = 64;

    bc::distributed::vector<bc::int_> distributed_vector(
        size, value, distributed_queue1
    );
    bc::distributed::vector<bc::int_> distributed_vector_copy1(
        distributed_vector
    );
    bc::distributed::vector<bc::int_> distributed_vector_copy2(
        distributed_vector, distributed_queue2
    );
    bc::distributed::vector<
        bc::int_,
        bc::distributed::default_weight_func, bc::pinned_allocator<bc::int_>
    > distributed_vector_copy3(
        distributed_vector, distributed_queue2
    );
    bc::distributed::vector<bc::int_> distributed_vector_copy4(
        distributed_vector, distributed_queue1
    );
    bc::distributed::vector<
        bc::int_,
        bc::distributed::default_weight_func, bc::pinned_allocator<bc::int_>
    > distributed_vector_copy5(
        distributed_vector, distributed_queue1
    );

    BOOST_CHECK(
        distributed_equal(distributed_vector, value, distributed_queue1)
    );
    BOOST_CHECK(
        distributed_equal(distributed_vector_copy1, value, distributed_queue1)
    );
    BOOST_CHECK(
        distributed_equal(distributed_vector_copy2, value, distributed_queue2)
    );
    BOOST_CHECK(
        distributed_equal(distributed_vector_copy3, value, distributed_queue2)
    );
    BOOST_CHECK(
        distributed_equal(distributed_vector_copy4, value, distributed_queue1)
    );
    BOOST_CHECK(
        distributed_equal(distributed_vector_copy5, value, distributed_queue1)
    );
}

BOOST_AUTO_TEST_CASE(at)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::distributed::vector<bc::uint_> distributed_vector(
        size_t(64), bc::uint_(64), distributed_queue
    );

    distributed_vector.begin(1).write(33, distributed_queue.get(1));
    distributed_queue.get(1).finish();

    BOOST_CHECK_EQUAL(
        distributed_vector.at(distributed_vector.part_start(1)),
        bc::uint_(33)
    );
    BOOST_CHECK_EQUAL(
        distributed_vector.at(distributed_vector.part_start(1) + 1),
        bc::uint_(64)
    );
    BOOST_CHECK_EQUAL(
        distributed_vector.at(distributed_vector.part_start(1) - 1),
        bc::uint_(64)
    );

    distributed_vector.at(distributed_vector.part_start(1)) = 55;
    BOOST_CHECK_EQUAL(
        *distributed_vector.begin(1),
        bc::uint_(55)
    );

    BOOST_CHECK_THROW(distributed_vector.at(64), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(subscript_operator)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::distributed::vector<bc::uint_> distributed_vector(
        size_t(64), bc::uint_(64), distributed_queue
    );

    distributed_vector.begin(1).write(bc::uint_(33), distributed_queue.get(1));
    distributed_queue.get(1).finish();

    BOOST_CHECK_EQUAL(
        distributed_vector[distributed_vector.part_start(1)],
        bc::uint_(33)
    );
    BOOST_CHECK_EQUAL(
        distributed_vector[distributed_vector.part_start(1) + 1],
        bc::uint_(64)
    );
    BOOST_CHECK_EQUAL(
        distributed_vector[distributed_vector.part_start(1) - 1],
        bc::uint_(64)
    );

    distributed_vector[distributed_vector.part_start(1)] = bc::uint_(55);
    BOOST_CHECK_EQUAL(
        *distributed_vector.begin(1),
        bc::uint_(55)
    );
}

BOOST_AUTO_TEST_CASE(swap)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue1 =
        get_distributed_queue(queue);
    // construct 2nd distributed::command_queue
    bc::distributed::command_queue distributed_queue2 =
        get_distributed_queue(queue, 2);

    bc::int_ value1 = -88;
    bc::int_ value2 = 99;
    size_t size1 = 64;
    size_t size2 = 48;

    bc::distributed::vector<bc::int_> distributed_vector1(
        size1, value1, distributed_queue1
    );
    bc::distributed::vector<bc::int_> distributed_vector2(
        size2, value2, distributed_queue2
    );

    BOOST_CHECK_EQUAL(distributed_vector1.size(), size1);
    BOOST_CHECK(
        distributed_equal(distributed_vector1, value1, distributed_queue1)
    );
    BOOST_CHECK_EQUAL(distributed_vector2.size(), size2);
    BOOST_CHECK(
        distributed_equal(distributed_vector2, value2, distributed_queue2)
    );
    distributed_queue1.finish();
    distributed_queue2.finish();

    distributed_vector1.swap(distributed_vector2);

    BOOST_CHECK_EQUAL(distributed_vector1.size(), size2);
    BOOST_CHECK(
        distributed_equal(distributed_vector1, value2, distributed_queue2)
    );
    BOOST_CHECK_EQUAL(distributed_vector2.size(), size1);
    BOOST_CHECK(
        distributed_equal(distributed_vector2, value1, distributed_queue1)
    );
}

BOOST_AUTO_TEST_SUITE_END()
