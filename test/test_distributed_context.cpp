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
    std::vector<bc::device> all_devices;

    const std::vector<bc::platform> &platforms = bc::system::platforms();
    for(size_t i = 0; i < platforms.size(); i++){
        const bc::platform &platform = platforms[i];
        std::vector<bc::device> platform_devices = platform.devices();

        // create a distributed context for devices in current platform
        bc::distributed::context ctx(platform_devices);

        // check context count
        BOOST_CHECK_EQUAL(ctx.size(), platform.device_count());

        all_devices.insert(all_devices.end(), platform_devices.begin(), platform_devices.end());
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
    bc::distributed::context ctx(contexts);

    BOOST_CHECK_EQUAL(ctx.size(), contexts.size());
    for(size_t i = 0; i < contexts.size(); i++) {
        BOOST_CHECK_EQUAL(ctx.get(i), contexts[i]);
    }
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

//BOOST_AUTO_TEST_CASE(test)
//{
//    bc::platform platform = Context::queue.get_context().get_device().platform();
//    // create a context for containing all devices in the platform
//    bc::context ctx(platform.devices());
//
//    for(size_t i = 0; i < platform.devices().size(); i++)
//    {
//        std::cout << platform.devices()[i].name() << std::endl;
//    }
//
//    bc::vector<bc::int_> vec(64, ctx);
//
//    bc::command_queue q0(ctx, platform.devices()[0]);
//    bc::command_queue q1(ctx, platform.devices()[1]);
//
//    bc::fill(vec.begin(), vec.begin() + 32, bc::int_(4), q0);
//    q0.finish();
//    bc::fill(vec.begin() + 32, vec.end(), bc::int_(3), q1);
//    q1.finish();
//
//    bc::fill(vec.begin(), vec.end(), bc::int_(5), q1);
//    q0.finish();
//
////    for(size_t i = 0; i < vec.size(); i++) {
////        std::cout << vec[i] << std::endl;
////    }
//}

BOOST_AUTO_TEST_SUITE_END()
