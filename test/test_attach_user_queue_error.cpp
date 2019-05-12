//---------------------------------------------------------------------------//
// Copyright (c) 2019 Anthony Chang <ac.chang@outlook.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAttachUserQueueError
#include <boost/test/unit_test.hpp>

#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>

namespace compute = boost::compute;

// For correct usage of setting up global default queue, see test_system.cpp
BOOST_AUTO_TEST_CASE(user_context_device_mismatch)
{
//! [queue_mismatch]
    compute::device user_device = compute::system::devices().front();
    compute::context user_context(user_device);
    compute::command_queue user_queue(user_context, user_device);

    // Don't call default_device() or default_context() before calling 
    // default_queue() if you wish to attach your command queue
    compute::system::default_context();

    BOOST_CHECK_THROW(
        compute::system::default_queue(user_queue),
        compute::set_default_queue_error);
//! [queue_mismatch]
}
