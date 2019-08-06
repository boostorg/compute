//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_TEST_DISTRIBUTED_QUEUE_SETUP_HPP
#define BOOST_COMPUTE_TEST_DISTRIBUTED_QUEUE_SETUP_HPP

#include <boost/compute/distributed/command_queue.hpp>

inline boost::compute::distributed::command_queue
get_distributed_queue(boost::compute::command_queue& queue,
                      size_t n = 1,
                      bool same_context = false)
{
    std::vector<boost::compute::command_queue> queues;
    queues.push_back(queue);
    for(size_t i = 0; i < n; i++) {
        if(same_context) {
            queues.push_back(
                boost::compute::command_queue(
                    queue.get_context(),
                    queue.get_device()
                )
            );
        }
        else {
            queues.push_back(
                boost::compute::command_queue(
                    boost::compute::context(queue.get_device()),
                    queue.get_device()
                )
            );
        }
    }

    return boost::compute::distributed::command_queue(queues);
}

#endif /* BOOST_COMPUTE_TEST_DISTRIBUTED_QUEUE_SETUP_HPP */
