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

float rand_float()
{
    return ((rand() / float(RAND_MAX)) - 0.5f) * 100000.0f;
}

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
    std::vector<float> host_vector(size);
    std::generate(host_vector.begin(), host_vector.end(), rand_float);

    // create vector on the device and copy the data
    boost::compute::vector<float> device_vector(size, context);
    boost::compute::copy(
        host_vector.begin(),
        host_vector.end(),
        device_vector.begin(),
        queue
    );

    // sort vector
    boost::compute::timer t(queue);
    boost::compute::sort(
        device_vector.begin(),
        device_vector.end(),
        queue
    );
    std::cout << "time: " << t.elapsed() / 1e6 << " ms" << std::endl;

    // verify vector is sorted
    if(!boost::compute::is_sorted(device_vector.begin(),
                                  device_vector.end(),
                                  queue)){
        std::cout << "ERROR: is_sorted() returned false" << std::endl;
        return -1;
    }

    return 0;
}
