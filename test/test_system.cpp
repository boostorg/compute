//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestSystem
#include <boost/test/unit_test.hpp>

#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(platform_count)
{
    BOOST_CHECK(compute::system::platform_count() >= 1);
}

BOOST_AUTO_TEST_CASE(device_count)
{
    BOOST_CHECK(compute::system::device_count() >= 1);
}

BOOST_AUTO_TEST_CASE(default_device)
{
    compute::device device = compute::system::default_device();
    BOOST_CHECK(device.id() != cl_device_id());
}

BOOST_AUTO_TEST_CASE(user_default_queue)
{
    compute::device device = compute::system::default_device();
    compute::context context = compute::context(device);

    compute::command_queue queue1(context, device);
    {
        compute::system::default_queue(queue1);
        compute::command_queue default_queue = compute::system::default_queue();
        BOOST_CHECK(queue1 == default_queue);
    }
#ifdef NDEBUG
    compute::command_queue queue2(context, device);
    {
        compute::system::default_queue(queue2); // no longer settable after first initialization
        compute::command_queue default_queue = compute::system::default_queue();
        BOOST_CHECK(queue2 != default_queue);
        BOOST_CHECK(queue1 == default_queue);
    }
#endif
}

BOOST_AUTO_TEST_CASE(find_device)
{
    compute::device device = compute::system::default_device();
    const std::string &name = device.name();
    BOOST_CHECK(compute::system::find_device(name).name() == device.name());
}
