//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

//[time_copy_example

#include <vector>
#include <cstdlib>
#include <iostream>

#include <boost/compute.hpp>

namespace compute = boost::compute;

int main()
{
    // get the default device
    compute::device gpu = compute::system::default_device();

    // create context for default device
    compute::context context(gpu);

    // create command queue with profiling enabled
    compute::command_queue queue(context,
                                 gpu,
                                 compute::command_queue::enable_profiling);

    // generate random data on the host
    std::vector<int> host_vector(100000);
    std::generate(host_vector.begin(), host_vector.end(), rand);

    // create a vector on the device
    compute::vector<int> device_vector(host_vector.size(), context);

    // create a timer
    compute::timer t(queue);

    // copy data from the host to the device
    compute::copy(host_vector.begin(),
                  host_vector.end(),
                  device_vector.begin(),
                  queue);

    // print elapsed time in milliseconds
    std::cout << "time: " << t.elapsed() / 1e6 << " ms" << std::endl;

    return 0;
}

//]
