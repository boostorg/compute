//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestDistributedTransform
#include <boost/test/unit_test.hpp>

#include <algorithm>

#include <boost/compute/algorithm.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/function.hpp>

#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/vector.hpp>
#include <boost/compute/distributed/transform.hpp>
#include <boost/compute/distributed/copy.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

#include "distributed_check_functions.hpp"
#include "distributed_queue_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(transform_async_int_abs)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::int_ data[] = { 1, -2, -3, -4, 5 };
    bc::distributed::vector<bc::int_> distributed_vector(
        data, data + 5, distributed_queue
    );
    distributed_queue.finish();

    std::vector<bc::int_> host(data, data + 5);
    BOOST_CHECK(
        distributed_equal(
            distributed_vector,
            host.begin(), host.end(),
            distributed_queue
        )
    );

    // transform
    std::vector<bc::event> events =
        bc::distributed::transform_async(
            distributed_vector,
            distributed_vector,
            bc::abs<bc::int_>(),
            distributed_queue
        );
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }

    bc::int_ expected_data[] = { 1, 2, 3, 4, 5 };
    std::vector<bc::int_> host_expected(expected_data, expected_data + 5);
    BOOST_CHECK(
        distributed_equal(
            distributed_vector,
            host_expected.begin(), host_expected.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(transform_float_custom_funtion)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::float_ data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    bc::distributed::vector<bc::float_> distributed_vector(
        data, data + 5, distributed_queue
    );
    distributed_queue.finish();

    BOOST_COMPUTE_FUNCTION(float, pow3add4, (float x),
    {
        return pow(x, 3.0f) + 4.0f;
    });

    // transform
    bc::distributed::transform(
        distributed_vector,
        distributed_vector,
        pow3add4,
        distributed_queue
    );
    distributed_queue.finish();

    BOOST_CHECK_CLOSE(bc::float_(distributed_vector[0]), 5.0f, 1e-4f);
    BOOST_CHECK_CLOSE(bc::float_(distributed_vector[1]), 12.0f, 1e-4f);
    BOOST_CHECK_CLOSE(bc::float_(distributed_vector[2]), 31.0f, 1e-4f);
    BOOST_CHECK_CLOSE(bc::float_(distributed_vector[3]), 68.0f, 1e-4f);
    BOOST_CHECK_CLOSE(bc::float_(distributed_vector[4]), 129.0f, 1e-4f);
}

BOOST_AUTO_TEST_CASE(transform_async_int_add)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::int_ data1[] = { 1, -2, -3, -4, 5 };
    bc::int_ data2[] = { -1, 2, 3, 4, 0 };
    std::vector<bc::int_> host1(data1, data1 + 5);
    std::vector<bc::int_> host2(data2, data2 + 5);

    bc::distributed::vector<bc::int_> distributed_vector1(
        data1, data1 + 5, distributed_queue
    );
    bc::distributed::vector<bc::int_> distributed_vector2(
        data2, data2 + 5, distributed_queue
    );
    bc::distributed::vector<bc::int_> distributed_vector3(
        size_t(5), distributed_queue
    );
    distributed_queue.finish();

    // add
    std::vector<bc::event> events =
        bc::distributed::transform_async(
            distributed_vector1,
            distributed_vector2,
            distributed_vector3,
            bc::plus<bc::int_>(),
            distributed_queue
        );
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }

    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            host1.begin(), host1.end(),
            distributed_queue
        )
    );
    BOOST_CHECK(
        distributed_equal(
            distributed_vector2,
            host2.begin(), host2.end(),
            distributed_queue
        )
    );

    bc::int_ expected_data_add[] = { 0, 0, 0, 0, 5 };
    std::vector<bc::int_> expected_add(
        expected_data_add, expected_data_add + 5
    );
    BOOST_CHECK(
        distributed_equal(
            distributed_vector3,
            expected_add.begin(), expected_add.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(transform_int_multiply)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::int_ data1[] = { 1, -2, -3, -4, 5 };
    bc::int_ data2[] = { -1, 2, 3, 4, 0 };
    std::vector<bc::int_> host1(data1, data1 + 5);
    std::vector<bc::int_> host2(data2, data2 + 5);

    bc::distributed::vector<bc::int_> distributed_vector1(
        data1, data1 + 5, distributed_queue
    );
    bc::distributed::vector<bc::int_> distributed_vector2(
        data2, data2 + 5, distributed_queue
    );
    bc::distributed::vector<bc::int_> distributed_vector3(
        size_t(5), distributed_queue
    );
    distributed_queue.finish();

    // multiply
    bc::distributed::transform(
        distributed_vector1,
        distributed_vector2,
        distributed_vector3,
        bc::multiplies<bc::int_>(),
        distributed_queue
    );

    BOOST_CHECK(
        distributed_equal(
            distributed_vector1,
            host1.begin(), host1.end(),
            distributed_queue
        )
    );
    BOOST_CHECK(
        distributed_equal(
            distributed_vector2,
            host2.begin(), host2.end(),
            distributed_queue
        )
    );

    bc::int_ expected_data_multiply[] = { -1, -4, -9, -16, 0 };
    std::vector<bc::int_> expected_multiply(
        expected_data_multiply, expected_data_multiply + 5
    );
    BOOST_CHECK(
        distributed_equal(
            distributed_vector3,
            expected_multiply.begin(), expected_multiply.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_SUITE_END()
