//---------------------------------------------------------------------------//
// Copyright (c) 2019 Anthony Chang <ac.chang@outlook.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestUserDefaultContext
#include <boost/test/unit_test.hpp>

#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>

namespace compute = boost::compute;

// For correct usage of setting up global default context, see test_system.cpp
BOOST_AUTO_TEST_CASE(user_context_device_mismatch)
{
    std::vector<compute::device> devices = compute::system::devices();
    compute::device user_device;
   
    for (std::vector<compute::device>::iterator it = devices.begin(); 
        it != devices.end(); 
        ++it)
    {
        if (it->type() & (compute::device::cpu | compute::device::gpu))
        {
            user_device = *it;
            break;
        }
    }
    
    compute::context user_context(user_device);

    // Don't call default_device() before calling default_context(&user_context)
    // if you wish to set your own default context
    compute::system::default_device(); 

    try 
    {
        compute::system::default_context(&user_context); 
    }
    catch (boost::compute::context_error& e) 
    {
        std::cout << e.what();
        BOOST_CHECK_EQUAL(
            std::string(e.what()), 
            std::string("Error: User CL context mismatches default device")
        );
        return;
    }
}
