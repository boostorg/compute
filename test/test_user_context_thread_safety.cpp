//---------------------------------------------------------------------------//
// Copyright (c) 2019 Anthony Chang <ac.chang@outlook.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestThreadSafeContext

#ifdef BOOST_COMPUTE_USE_CPP11
  #include <thread>
  using std::thread;
#else
  #include <boost/thread.hpp>
  using boost::thread;
#endif 

#include <boost/test/unit_test.hpp>

#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>

namespace compute = boost::compute;

void set_default_context_workers_threads(
    int id,
    const compute::device& user_device, 
    compute::context* contexts)
{
    compute::context user_context(user_device);
    contexts[id] = compute::system::default_context(&user_context);
}

compute::device select_user_device()
{
    std::vector<compute::device> devices = compute::system::devices();

    for (std::vector<compute::device>::iterator it = devices.begin();
         it != devices.end();
         ++it)
    {
        if (it->type() & (compute::device::cpu | compute::device::gpu))
        {
            return *it;
        }
    }

    BOOST_REQUIRE(false); // should've found some CL device
    return compute::device();
}

BOOST_AUTO_TEST_CASE(user_default_context_thread_safety)
{
#ifdef BOOST_COMPUTE_THREAD_SAFE
    const int num_threads = 16;
    
    compute::device user_device = select_user_device();
    compute::context contexts[num_threads];
    thread* threads[num_threads];
    for (int i = 0; i < num_threads; i++)
    {
        threads[i] = new thread(
            set_default_context_workers_threads,
            i, user_device, contexts
        );
    }

    for (int i = 0; i < num_threads; i++)
    {
        threads[i]->join();
        delete threads[i];
    }

    for (int i = 1; i < num_threads; i++)
    {
        BOOST_CHECK_EQUAL(contexts[0].get(), contexts[i].get());
    }
#endif
}
