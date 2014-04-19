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

#include <boost/config.hpp>

#if !defined(BOOST_NO_CXX11_HDR_FUTURE) && !defined(BOOST_NO_0X_HDR_FUTURE)
#include <future>
#endif // BOOST_NO_CXX11_HDR_FUTURE

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

#if !defined(BOOST_NO_CXX11_LAMBDAS) && !defined(BOOST_NO_LAMBDAS)
BOOST_AUTO_TEST_CASE(lambda_callback)
{
    bool lambda_invoked = false;
    boost::compute::event marker = queue.enqueue_marker();
    marker.set_callback([&lambda_invoked](){ lambda_invoked = true; });
    queue.finish();
    BOOST_CHECK_EQUAL(lambda_invoked, true);
}
#endif // BOOST_NO_CXX11_LAMBDAS

#if !defined(BOOST_NO_CXX11_HDR_FUTURE) && !defined(BOOST_NO_0X_HDR_FUTURE)
void BOOST_COMPUTE_CL_CALLBACK
event_promise_fulfiller_callback(cl_event event, cl_int status, void *user_data)
{
    auto *promise = static_cast<std::promise<void> *>(user_data);
    promise->set_value();
    delete promise;
}

BOOST_AUTO_TEST_CASE(event_to_std_future)
{
    std::vector<float> vector(1000, 3.14f);
    boost::compute::buffer buffer(context, 1000 * sizeof(float));
    auto event = queue.enqueue_write_buffer_async(
        buffer, 0, 1000 * sizeof(float), vector.data()
    );
    auto *promise = new std::promise<void>;
    std::future<void> future = promise->get_future();
    event.set_callback(event_promise_fulfiller_callback, CL_COMPLETE, promise);
    future.wait();
}
#endif // BOOST_NO_CXX11_HDR_FUTURE
#endif // CL_VERSION_1_1

BOOST_AUTO_TEST_SUITE_END()
