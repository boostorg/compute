//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestDistributedScan
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
#include <boost/compute/distributed/exclusive_scan.hpp>
#include <boost/compute/distributed/copy.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

#include "distributed_check_functions.hpp"
#include "distributed_queue_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(exclusive_scan_int)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue, 4);

    std::vector<bc::int_> data(size_t(128));
    for(size_t i = 0; i < data.size(); i++) {
        data[i] = i;
    }

    bc::distributed::vector<bc::int_> distributed_input(
        data.begin(), data.end(), distributed_queue
    );
    bc::distributed::vector<bc::int_> distributed_result(
        data.size(), distributed_queue
    );
    distributed_queue.finish();

    BOOST_CHECK(
        distributed_equal(
            distributed_input,
            data.begin(), data.end(),
            distributed_queue
        )
    );

    bc::distributed::exclusive_scan(
        distributed_input,
        distributed_result,
        bc::int_(10),
        distributed_queue
    );
    distributed_queue.finish();

    bc::vector<bc::int_> device_input(data.begin(), data.end(), queue);
    bc::vector<bc::int_> device_expected(data.size(), context);
    std::vector<bc::int_> host_expected(device_expected.size());
    bc::exclusive_scan(
            device_input.begin(),
            device_input.end(),
            device_expected.begin(),
            bc::int_(10),
            queue
    );
    bc::copy(
        device_expected.begin(),
        device_expected.end(),
        host_expected.begin(),
        queue
    );
    queue.finish();

    BOOST_CHECK(
        distributed_equal(
            distributed_input,
            data.begin(), data.end(),
            distributed_queue
        )
    );
    BOOST_CHECK(
        distributed_equal(
            distributed_result,
            host_expected.begin(), host_expected.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_CASE(exclusive_scan_custom_function_int)
{
    // construct distributed::command_queue
    bc::distributed::command_queue distributed_queue =
        get_distributed_queue(queue, 3);

    BOOST_COMPUTE_FUNCTION(bc::int_, custom_sum, (bc::int_ x, bc::int_ y),
    {
        return x + y;
    });

    std::vector<bc::int_> data(size_t(128));
    for(size_t i = 0; i < data.size(); i++) {
        data[i] = i;
    }

    bc::distributed::vector<bc::int_> distributed_input(
        data.begin(), data.end(), distributed_queue
    );
    distributed_queue.finish();

    BOOST_CHECK(
        distributed_equal(
            distributed_input,
            data.begin(), data.end(),
            distributed_queue
        )
    );

    bc::distributed::exclusive_scan(
        distributed_input,
        distributed_input,
        bc::int_(10),
        custom_sum,
        distributed_queue
    );
    distributed_queue.finish();

    bc::vector<bc::int_> device_input(data.begin(), data.end(), queue);
    bc::vector<bc::int_> device_expected(data.size(), context);
    std::vector<bc::int_> host_expected(device_expected.size());
    bc::exclusive_scan(
            device_input.begin(),
            device_input.end(),
            device_expected.begin(),
            bc::int_(10),
            queue
    );
    bc::copy(
        device_expected.begin(),
        device_expected.end(),
        host_expected.begin(),
        queue
    );
    queue.finish();
    BOOST_CHECK(
        distributed_equal(
            distributed_input,
            host_expected.begin(), host_expected.end(),
            distributed_queue
        )
    );
}

BOOST_AUTO_TEST_SUITE_END()
