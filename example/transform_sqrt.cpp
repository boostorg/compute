//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

//[transform_sqrt_example

#include <vector>
#include <algorithm>
#include <boost/compute.hpp>

namespace compute = boost::compute;

int main()
{
    // generate random data on the host
    std::vector<float> host_vector(10000);
    std::generate(host_vector.begin(), host_vector.end(), rand);

    // create a vector on the device and transfer data from the host
    compute::vector<float> device_vector = host_vector;

    // calculate sqrt of each element in-place
    compute::transform(device_vector.begin(),
                       device_vector.end(),
                       device_vector.begin(),
                       compute::sqrt<float>());

    // copy values back to the host
    compute::copy(device_vector.begin(),
                  device_vector.end(),
                  host_vector.begin());

    return 0;
}

//]
