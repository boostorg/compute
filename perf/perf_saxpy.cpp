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

#include <boost/compute/lambda.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>

#include "perf.hpp"

float rand_float()
{
    return (float(rand()) / float(RAND_MAX)) * 1000.f;
}

// y <- alpha * x + y
void serial_saxpy(size_t n, float alpha, const float *x, float *y)
{
    for(size_t i = 0; i < n; i++){
        y[i] = alpha * x[i] + y[i];
    }
}

int main(int argc, char *argv[])
{
    perf_parse_args(argc, argv);

    using boost::compute::lambda::_1;
    using boost::compute::lambda::_2;

    std::cout << "size: " << PERF_N << std::endl;

    float alpha = 2.5f;

    // setup context and queue for the default device
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);
    std::cout << "device: " << device.name() << std::endl;

    // create vector of random numbers on the host
    std::vector<float> host_x(PERF_N);
    std::vector<float> host_y(PERF_N);
    std::generate(host_x.begin(), host_x.end(), rand_float);
    std::generate(host_y.begin(), host_y.end(), rand_float);

    // create vector on the device and copy the data
    boost::compute::vector<float> device_x(host_x.begin(), host_x.end(), queue);
    boost::compute::vector<float> device_y(host_y.begin(), host_y.end(), queue);

    perf_timer t;
    for(size_t trial = 0; trial < PERF_TRIALS; trial++){
        boost::compute::copy(host_x.begin(), host_x.end(), device_x.begin(), queue);
        boost::compute::copy(host_y.begin(), host_y.end(), device_y.begin(), queue);

        t.start();
        boost::compute::transform(
            device_x.begin(),
            device_x.end(),
            device_y.begin(),
            device_y.begin(),
            alpha * _1 + _2,
            queue
        );
        queue.finish();
        t.stop();
    }
    std::cout << "time: " << t.min_time() / 1e6 << " ms" << std::endl;

    // perform saxpy on host
    serial_saxpy(PERF_N, alpha, &host_x[0], &host_y[0]);

    // copy device_y to host_x
    boost::compute::copy(device_y.begin(), device_y.end(), host_x.begin(), queue);

    for(size_t i = 0; i < PERF_N; i++){
        float host_value = host_y[i];
        float device_value = host_x[i];

        if(std::abs(device_value - host_value) > 1e-3){
            std::cout << "ERROR: "
                      << "value at " << i << " "
                      << "device_value (" << device_value << ") "
                      << "!= "
                      << "host_value (" << host_value << ")"
                      << std::endl;
            return -1;
        }
    }

    return 0;
}
