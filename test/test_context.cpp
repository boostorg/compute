//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestContext
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>

BOOST_AUTO_TEST_CASE(construct_from_cl_context)
{
    boost::compute::device device = boost::compute::system::default_device();
    cl_device_id id = device.id();

    // create cl_context
    cl_context ctx = clCreateContext(0, 1, &id, 0, 0, 0);
    BOOST_VERIFY(ctx);

    // create boost::compute::context
    boost::compute::context context(ctx);

    // check context
    BOOST_CHECK(cl_context(context) == ctx);

    // cleanup cl_context
    clReleaseContext(ctx);
}

#ifndef BOOST_COMPUTE_NO_RVALUE_REFERENCES
BOOST_AUTO_TEST_CASE(move_constructor)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context1(device);
    BOOST_VERIFY(cl_context(context1) != cl_context());

    boost::compute::context context2(std::move(context1));
    BOOST_VERIFY(cl_context(context2) != cl_context());
    BOOST_VERIFY(cl_context(context1) == cl_context());
}
#endif // BOOST_COMPUTE_NO_RVALUE_REFERENCES
