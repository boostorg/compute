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
#include <boost/compute/detail/timer.hpp>

namespace compute = boost::compute;

using compute::float2_;

float rand_float()
{
    return (float(rand()) / float(RAND_MAX)) * 1000.f;
}

void serial_cartesian_to_polar(const float *input, size_t n, float *output)
{
    for(size_t i = 0; i < n; i++){
        float x = input[i*2+0];
        float y = input[i*2+1];

        float magnitude = std::sqrt(x*x + y*y);
        float angle = std::atan2(y, x) * 180.f / M_PI;

        output[i*2+0] = magnitude;
        output[i*2+1] = angle;
    }
}

void serial_polar_to_cartesian(const float *input, size_t n, float *output)
{
    for(size_t i = 0; i < n; i++){
        float magnitude = input[i*2+0];
        float angle = input[i*2+1];

        float x = magnitude * cos(angle);
        float y = magnitude * sin(angle);

        output[i*2+0] = x;
        output[i*2+1] = y;
    }
}

// converts from cartesian coordinates (x, y) to polar coordinates (magnitude, angle)
BOOST_COMPUTE_FUNCTION(float2_, cartesian_to_polar, (float2_),
{
    float x = _1.x;
    float y = _1.y;

    float magnitude = sqrt(x*x + y*y);
    float angle = atan2(y, x) * 180.f / M_PI;

    return (float2)(magnitude, angle);
});

// converts from polar coordinates (magnitude, angle) to cartesian coordinates (x, y)
BOOST_COMPUTE_FUNCTION(float2_, polar_to_cartesian, (float2_),
{
    float magnitude = _1.x;
    float angle = _1.y;

    float x = magnitude * cos(angle);
    float y = magnitude * sin(angle);

    return (float2)(x, y)
});

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

    // create vector of random numbers on the host
    std::vector<float> host_vector(size*2);
    std::generate(host_vector.begin(), host_vector.end(), rand_float);

    // create vector on the device and copy the data
    compute::vector<float2_> device_vector(size, context);
    compute::copy_n(
        reinterpret_cast<float2_ *>(&host_vector[0]),
        size,
        device_vector.begin(),
        queue
    );

    compute::detail::timer t;
    compute::transform(
        device_vector.begin(),
        device_vector.end(),
        device_vector.begin(),
        cartesian_to_polar,
        queue
    );
    queue.finish();
    std::cout << "time: " << t.elapsed() << " ms" << std::endl;

    // perform saxpy on host
    t.start();
    serial_cartesian_to_polar(&host_vector[0], size, &host_vector[0]);
    std::cout << "host time: " << t.elapsed() << " ms" << std::endl;

    std::vector<float> device_data(size*2);
    compute::copy(
        device_vector.begin(),
        device_vector.end(),
        reinterpret_cast<float2_ *>(&device_data[0]),
        queue
    );

    for(size_t i = 0; i < size; i++){
        float host_value = host_vector[i];
        float device_value = device_data[i];

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
