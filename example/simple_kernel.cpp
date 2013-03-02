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

// this example demonstrates how to use the Boost.Compute classes to
// setup and run a simple vector addition kernel on the GPU
int main()
{
    // get the default GPU device
    boost::compute::device gpu =
        boost::compute::system::default_gpu_device();

    // create a context for the device
    boost::compute::context context(gpu);

    // setup input arrays
    float a[] = { 1, 2, 3, 4 };
    float b[] = { 5, 6, 7, 8 };

    // make space for the output
    float c[] = { 0, 0, 0, 0 };

    // create memory buffers for the input and output
    boost::compute::buffer buffer_a(context, 4 * sizeof(float));
    boost::compute::buffer buffer_b(context, 4 * sizeof(float));
    boost::compute::buffer buffer_c(context, 4 * sizeof(float));

    // source code for the add kernel
    const char source[] =
        "__kernel void add(__global const float *a,"
        "                  __global const float *b,"
        "                  __global float *c)"
        "{"
        "    const uint i = get_global_id(0);"
        "    c[i] = a[i] + b[i];"
        "}";

    // create the program with the source
    boost::compute::program program =
        boost::compute::program::create_with_source(source, context);

    // compile the program
    program.build();

    // create the kernel
    boost::compute::kernel kernel(program, "add");

    // set the kernel arguments
    kernel.set_arg(0, buffer_a);
    kernel.set_arg(1, buffer_b);
    kernel.set_arg(2, buffer_c);

    // create a command queue
    boost::compute::command_queue queue(context, gpu);

    // write the data from 'a' and 'b' to the device
    queue.enqueue_write_buffer(buffer_a, a);
    queue.enqueue_write_buffer(buffer_b, b);

    // run the add kernel
    queue.enqueue_1d_range_kernel(kernel, 0, 4);

    // transfer results back to the host array 'c'
    queue.enqueue_read_buffer(buffer_c, c);

    // print out results in 'c'
    std::cout << "c: [" << c[0] << ", "
                        << c[1] << ", "
                        << c[2] << ", "
                        << c[3] << "]" << std::endl;

    return 0;
}
