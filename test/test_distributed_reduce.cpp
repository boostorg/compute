//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestDistributedReduce
#include <boost/test/unit_test.hpp>

#include <algorithm>

#include <boost/compute/algorithm.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/container/vector.hpp>

#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/vector.hpp>
#include <boost/compute/distributed/reduce.hpp>
#include <boost/compute/distributed/copy.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

#include "distributed_check_functions.hpp"
#include "distributed_queue_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(reduce_int_to_host)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::int_ data[] = { 5, 1, 9, 17, 13 };
    bc::distributed::vector<bc::int_> distributed_vector(
        data, data + 5, distributed_queue
    );
    distributed_queue.finish();

    bc::int_ sum;
    bc::distributed::reduce(
        distributed_vector,
        &sum,
        bc::plus<bc::int_>(),
        distributed_queue
    );
    BOOST_CHECK_EQUAL(sum, 45);

    bc::int_ product;
    bc::distributed::reduce(
        distributed_vector,
        &product,
        bc::multiplies<bc::int_>(),
        distributed_queue
    );
    BOOST_CHECK_EQUAL(product, 9945);

    bc::int_ min_value;
    bc::distributed::reduce(
        distributed_vector,
        &min_value,
        bc::min<bc::int_>(),
        distributed_queue
    );
    BOOST_CHECK_EQUAL(min_value, 1);

    bc::int_ max_value;
    bc::distributed::reduce(
        distributed_vector,
        &max_value,
        bc::max<bc::int_>(),
        distributed_queue
    );
    BOOST_CHECK_EQUAL(max_value, 17);
}

BOOST_AUTO_TEST_CASE(reduce_int_to_device)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::int_ data[] = { 1, 5, 9, 13, 17 };
    bc::distributed::vector<bc::int_> distributed_vector(
        data, data + 5, distributed_queue
    );
    distributed_queue.finish();

    bc::vector<bc::int_> result1(1, distributed_queue.get_context(0));
    bc::distributed::reduce(
        distributed_vector,
        result1.begin(),
        bc::plus<bc::int_>(),
        distributed_queue
    );
    BOOST_CHECK_EQUAL(result1.begin().read(queue), 45);

    bc::vector<bc::int_> result2(1, distributed_queue.get_context(1));
    bc::distributed::reduce(
        distributed_vector,
        result2.begin(),
        bc::multiplies<bc::int_>(),
        distributed_queue
    );
    BOOST_CHECK_EQUAL(result2.begin().read(distributed_queue.get(1)), 9945);
}

BOOST_AUTO_TEST_CASE(reduce_int_to_device_one_context)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue, 1, true);

    bc::distributed::vector<bc::int_> distributed_vector(
        size_t(1024), bc::int_(1), distributed_queue
    );
    distributed_vector[0] = 2;
    distributed_queue.finish();

    bc::vector<bc::int_> result1(1, context);
    bc::distributed::reduce(
        distributed_vector,
        result1.begin(),
        bc::plus<bc::int_>(),
        distributed_queue
    );
    BOOST_CHECK_EQUAL(result1.begin().read(queue), 1025);

    bc::vector<bc::int_> result2(1, context);
    bc::distributed::reduce(
        distributed_vector,
        result2.begin(),
        bc::multiplies<bc::int_>(),
        distributed_queue
    );
    distributed_queue.finish();
    BOOST_CHECK_EQUAL(result2.begin().read(distributed_queue.get(0)), 2);
}

BOOST_AUTO_TEST_CASE(reduce_int_custom_function)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue);

    bc::distributed::vector<bc::int_> distributed_vector(
        size_t(34), bc::int_(2), distributed_queue
    );
    distributed_queue.finish();

    BOOST_COMPUTE_FUNCTION(bc::float_, custom_sum, (bc::float_ x, bc::float_ y),
    {
        return x + y;
    });


    bc::float_ sum;
    bc::distributed::reduce(
        distributed_vector,
        &sum,
        custom_sum,
        distributed_queue
    );
    BOOST_CHECK_CLOSE(sum, bc::float_(68), 0.01);
}

BOOST_AUTO_TEST_SUITE_END()
