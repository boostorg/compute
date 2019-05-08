//---------------------------------------------------------------------------//
// Copyright (c) 2019 Anthony Chang <ac.chang@outlook.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAttachUserQueueThreadSafety

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

void attach_user_queue_worker_threads(
    int id,
    const compute::context& user_context,
    const compute::device& user_device,
    compute::command_queue* queues)
{
    compute::command_queue user_queue(user_context, user_device);
    queues[id] = compute::system::default_queue(user_queue);
}

BOOST_AUTO_TEST_CASE(user_default_context_thread_safety)
{
#if defined(BOOST_COMPUTE_THREAD_SAFE) && defined(NDEBUG)
    const int num_threads = 16;

    compute::command_queue queues[num_threads];
    thread* threads[num_threads];
    
    compute::device user_device = compute::system::devices().front();
    compute::context user_context(user_device);

    for (int i = 0; i < num_threads; i++)
    {
        threads[i] = new thread(
            attach_user_queue_worker_threads,
            i, 
            user_context, 
            user_device,
            queues
        );
    }

    for (int i = 0; i < num_threads; i++)
    {
        threads[i]->join();
        delete threads[i];
    }

    BOOST_CHECK_EQUAL(compute::system::default_context().get(), user_context.get());
    BOOST_CHECK_EQUAL(compute::system::default_device().get(), user_device.get());
    for (int i = 1; i < num_threads; i++)
    {
        BOOST_CHECK_EQUAL(queues[0].get(), queues[i].get());
    }
#endif // defined(BOOST_COMPUTE_THREAD_SAFE) && defined(NDEBUG)
}
