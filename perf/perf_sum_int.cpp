//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#include <algorithm>
#include <iostream>
#include <vector>

#include <boost/compute.hpp>

int main(int argc, char *argv[])
{
    size_t size = 1000;
    if(argc >= 2){
        size = boost::lexical_cast<size_t>(argv[1]);
    }

    std::cout << "size: " << size << std::endl;

    // setup context and queue for the default device
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(
        context,
        device,
        boost::compute::command_queue::enable_profiling
    );

    // create vector of random numbers on the host
    std::vector<int> host_vector(size);
    std::generate(host_vector.begin(), host_vector.end(), rand);

    // create vector on the device and copy the data
    boost::compute::vector<int> device_vector(size, context);
    boost::compute::copy(
        host_vector.begin(),
        host_vector.end(),
        device_vector.begin(),
        queue
    );

    // sum vector
    boost::compute::timer t(queue);
    int sum =
        boost::compute::accumulate(device_vector.begin(),
                                   device_vector.end(),
                                   int(0),
                                   queue);
    std::cout << "time: " << t.elapsed() / 1e6 << " ms" << std::endl;

    // verify sum is correct
    int host_sum = std::accumulate(host_vector.begin(),
                                   host_vector.end(),
                                   int(0));
    if(sum != host_sum){
        std::cout << "ERROR: "
                  << "device_sum (" << sum << ") "
                  << "!= "
                  << "host_sum (" << host_sum << ")"
                  << std::endl;
        return -1;
    }

    return 0;
}
