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

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random/mersenne_twister.hpp>
#include <boost/compute/detail/timer.hpp>

namespace compute = boost::compute;

int main(int argc, char *argv[])
{
    size_t size = 1000;
    if(argc >= 2){
        size = boost::lexical_cast<size_t>(argv[1]);
    }

    std::cout << "size: " << size << std::endl;

    // setup context and queue for the default device
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    // create vector on the device
    compute::vector<unsigned int> vector(size, context);

    // create mersenne twister engine
    compute::mt19937 rng(context);

    // generate random numbers
    compute::detail::timer t;
    rng.fill(vector.begin(), vector.end(), queue);
    queue.finish();
    std::cout << "time: " << t.elapsed() << " ms" << std::endl;

    return 0;
}
