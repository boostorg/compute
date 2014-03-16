//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestEvent
#include <boost/test/unit_test.hpp>

#include <boost/compute/event.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(null_event)
{
    boost::compute::event null;
    BOOST_CHECK(null.get() == cl_event());
}

#ifdef CL_VERSION_1_1
static bool callback_invoked = false;

static void BOOST_COMPUTE_CL_CALLBACK
callback(cl_event event, cl_int status, void *user_data)
{
    callback_invoked = true;
}

BOOST_AUTO_TEST_CASE(event_callback)
{
    BOOST_CHECK_EQUAL(callback_invoked, false);
    boost::compute::event marker = queue.enqueue_marker();
    marker.set_callback(callback);
    queue.finish();
    BOOST_CHECK_EQUAL(callback_invoked, true);
}
#endif // CL_VERSION_1_1

BOOST_AUTO_TEST_SUITE_END()
