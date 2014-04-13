//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
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

#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/random/mersenne_twister.hpp>

#include "perf.hpp"

namespace compute = boost::compute;

int main(int argc, char *argv[])
{
    perf_parse_args(argc, argv);
    std::cout << "size: " << PERF_N << std::endl;

    // setup context and queue for the default device
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    // create vector on the device
    compute::vector<unsigned int> vector(PERF_N, context);

    // create mersenne twister engine
    compute::mt19937 rng(queue);

    // generate random numbers
    perf_timer t;
    t.start();
    rng.generate(vector.begin(), vector.end(), queue);
    queue.finish();
    t.stop();
    std::cout << "time: " << t.last_time() / 1e6 << " ms" << std::endl;

    return 0;
}
