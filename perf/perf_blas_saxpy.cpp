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
#include <boost/compute/blas/axpy.hpp>
#include <boost/compute/detail/timer.hpp>

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
    size_t size = 1000;
    if(argc >= 2){
        size = boost::lexical_cast<size_t>(argv[1]);
    }

    std::cout << "size: " << size << std::endl;

    // setup context and queue for the default device
    boost::compute::device device = boost::compute::system::default_device();

    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    // create vector of random numbers on the host
    std::vector<float> host_x(size);
    std::vector<float> host_y(size);
    std::generate(host_x.begin(), host_x.end(), rand_float);
    std::generate(host_y.begin(), host_y.end(), rand_float);

    // create vector on the device and copy the data
    boost::compute::vector<float> device_x(host_x.begin(), host_x.end(), context);
    boost::compute::vector<float> device_y(host_y.begin(), host_y.end(), context);

    boost::compute::detail::timer t;
    boost::compute::blas::axpy(size, 2.5f, &device_x[0], 1, &device_y[0], 1, queue);
    queue.finish();
    std::cout << "time: " << t.elapsed() << " ms" << std::endl;

    // perform saxpy on host
    serial_saxpy(size, 2.5f, &host_x[0], &host_y[0]);

    // copy device_y to host_x
    boost::compute::copy(device_y.begin(), device_y.end(), host_x.begin(), queue);

    for(size_t i = 0; i < size; i++){
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
