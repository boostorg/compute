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

#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>

BOOST_AUTO_TEST_CASE(platform_count)
{
    BOOST_CHECK(boost::compute::system::platform_count() >= 1);
}

BOOST_AUTO_TEST_CASE(device_count)
{
    BOOST_CHECK(boost::compute::system::device_count() >= 1);
}

BOOST_AUTO_TEST_CASE(default_device)
{
    boost::compute::device device = boost::compute::system::default_device();
    BOOST_CHECK(device.id() != cl_device_id());
}

BOOST_AUTO_TEST_CASE(user_default_context)
{
    boost::compute::device device = boost::compute::system::default_device();

    boost::compute::context context1(device);
    {
        boost::compute::system::default_context(&context1);
        boost::compute::context default_context = boost::compute::system::default_context();
        BOOST_ASSERT(context1 == default_context);
    }
    boost::compute::context context2(device);
    {
        boost::compute::system::default_context(&context2); // no longer settable
        boost::compute::context default_context = boost::compute::system::default_context();
        BOOST_ASSERT(context2 != default_context);
        BOOST_ASSERT(context1 == default_context);
    }
}

BOOST_AUTO_TEST_CASE(find_device)
{
    boost::compute::device device = boost::compute::system::default_device();
    const std::string &name = device.name();
    BOOST_CHECK(boost::compute::system::find_device(name).name() == device.name());
}
