//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#include <iostream>
#include <boost/compute.hpp>

namespace compute = boost::compute;

int main()
{
    compute::device gpu = compute::system::default_device();
    compute::context context(gpu);
    compute::command_queue queue(context, gpu);

    std::cout << "device: " << gpu.name() << std::endl;

    using compute::uint_;

    size_t n = 2500;

    // generate random numbers
    compute::default_random_engine rng(context);
    compute::vector<uint_> vector(n * 2, context);
    rng.fill(vector.begin(), vector.end(), queue);

    // compile program
    const char source[] =
        "__kernel void count_in_circle(__global uint *count,\n"
        "                              __global uint *random_values)\n"
        "{\n"
        "    const uint i = get_global_id(0);\n"
        "    const uint ix = random_values[i*2+0];\n"
        "    const uint iy = random_values[i*2+1];\n"
        "    const float x = ix / (float) UINT_MAX - 1;\n"
        "    const float y = iy / (float) UINT_MAX - 1;\n"
        "    if((x*x + y*y) < 1.0f) atomic_inc(count);\n"
        "}\n";

    compute::program program =
        compute::program::create_with_source(source, context);
    program.build();

    compute::vector<uint_> count(1, context);
    count[0] = 0;

    compute::kernel count_kernel(program, "count_in_circle");
    count_kernel.set_arg(0, count.get_buffer());
    count_kernel.set_arg(1, vector.get_buffer());

    // execute kernel
    queue.enqueue_1d_range_kernel(count_kernel, 0, n);
    queue.finish();

    float total = float(uint_(count[0]));

    std::cout << "count: " << uint_(count[0]) << " / " << n << std::endl;
    std::cout << "ratio: " << total / float(n) << std::endl;
    std::cout << "pi = " << (total / float(n)) * 4.0f << std::endl;

    return 0;
}
