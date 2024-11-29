//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestDistributedContext
#include <boost/test/unit_test.hpp>

#include <boost/compute/algorithm.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/distributed/context.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(construct_from_devices)
{
    std::vector<std::vector<bc::device> > all_devices;

    const std::vector<bc::platform> &platforms = bc::system::platforms();
    for(size_t i = 0; i < platforms.size(); i++){
        const bc::platform &platform = platforms[i];

        std::vector<bc::device> platform_devices = platform.devices();
        std::vector<cl_context_properties*> properties(platform_devices.size(), 0);

        // create a distributed context for devices in current platform
        bc::distributed::context ctx1(platform_devices);
        bc::distributed::context ctx2(platform_devices, properties);

        // check context count
        BOOST_CHECK_EQUAL(ctx1.size(), platform_devices.size());
        BOOST_CHECK_EQUAL(ctx2.size(), platform_devices.size());

        all_devices.push_back(platform_devices);
    }

    // create a distributed context for devices in current platform
    std::vector<cl_context_properties*> properties(all_devices.size(), 0);
    bc::distributed::context ctx(all_devices, properties);
    BOOST_CHECK_EQUAL(ctx.size(), all_devices.size());
}

BOOST_AUTO_TEST_CASE(construct_from_contexts)
{
    std::vector<bc::context> contexts;

    const std::vector<bc::platform> &platforms = bc::system::platforms();
    for(size_t i = 0; i < platforms.size(); i++){
        const bc::platform &platform = platforms[i];

        // create a context for containing all devices in the platform
        bc::context ctx(platform.devices());
        contexts.push_back(ctx);
    }

    bc::distributed::context ctx1(contexts);
    bc::distributed::context ctx2(contexts.begin(), contexts.end());

    BOOST_CHECK_EQUAL(ctx1.size(), contexts.size());
    BOOST_CHECK_EQUAL(ctx2.size(), contexts.size());
    for(size_t i = 0; i < contexts.size(); i++) {
        BOOST_CHECK_EQUAL(ctx1.get(i), contexts[i]);
        BOOST_CHECK_EQUAL(ctx2.get(i), contexts[i]);
    }
}

BOOST_AUTO_TEST_CASE(construct_from_context)
{
    bc::distributed::context ctx(context);
    BOOST_CHECK_EQUAL(ctx.size(), 1);
    BOOST_CHECK_EQUAL(ctx.get(0), context);
}

BOOST_AUTO_TEST_CASE(copy_ctor)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);
    bc::distributed::context distributed_context1(contexts);
    bc::distributed::context distributed_context2(distributed_context1);
    BOOST_CHECK(
        distributed_context1 == distributed_context2
    );
}

BOOST_AUTO_TEST_CASE(assign_operator)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);
    bc::distributed::context distributed_context1(contexts);
    bc::distributed::context distributed_context2 = distributed_context1;
    BOOST_CHECK(
        distributed_context1 == distributed_context2
    );
}

BOOST_AUTO_TEST_CASE(equality_operator)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);
    bc::distributed::context distributed_context1(contexts);
    bc::distributed::context distributed_context2(contexts);

    contexts.push_back(
        bc::context(queue.get_device())
    );
    bc::distributed::context distributed_context3(contexts);

    BOOST_CHECK(distributed_context1 == distributed_context2);
    BOOST_CHECK(distributed_context2 == distributed_context1);

    BOOST_CHECK(distributed_context1 != distributed_context3);
    BOOST_CHECK(distributed_context3 != distributed_context1);

    BOOST_CHECK(distributed_context2 != distributed_context3);
    BOOST_CHECK(distributed_context3 != distributed_context2);
}

BOOST_AUTO_TEST_CASE(get_info)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);
    bc::distributed::context distributed_context(contexts);

    BOOST_CHECK(
        distributed_context.get_info<std::vector<cl_device_id> >(0, CL_CONTEXT_DEVICES)
            == context.get_info<std::vector<cl_device_id> >(CL_CONTEXT_DEVICES)
    );
}

BOOST_AUTO_TEST_CASE(get_context)
{
    std::vector<bc::context> contexts;
    contexts.push_back(context);
    contexts.push_back(context);
    contexts.push_back(bc::context(context.get_device()));
    bc::distributed::context distributed_context(contexts);

    BOOST_CHECK_EQUAL(distributed_context.get(0), context);
    BOOST_CHECK_EQUAL(distributed_context.get(1), context);
    for(size_t i = 0; i < distributed_context.size(); i++) {
        BOOST_CHECK_EQUAL(distributed_context.get(i), contexts[i]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
