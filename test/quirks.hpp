//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_TEST_QUIRKS_HPP
#define BOOST_COMPUTE_TEST_QUIRKS_HPP

#include <boost/compute/device.hpp>
#include <boost/compute/platform.hpp>

// this file contains functions which check for 'quirks' or buggy
// behavior in OpenCL implementations. this allows us to skip certain
// tests when running on buggy platforms.

// AMD platforms have a bug when using struct assignment. this affects
// algorithms like fill() when used with pairs/tuples.
//
// see: http://devgurus.amd.com/thread/166622
inline bool bug_in_struct_assignment(const boost::compute::device &device)
{
    boost::compute::platform platform(
        device.get_info<cl_platform_id>(CL_DEVICE_PLATFORM)
    );

    if(platform.vendor() == "Advanced Micro Devices, Inc."){
        return true;
    }

    return false;
}

#endif // BOOST_COMPUTE_TEST_QUIRKS_HPP
