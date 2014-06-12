//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Mageswaran.D <mageswaran1989@gmail.com>
// Reference Paper: An Introduction to the OpenCL Programming Model
//                 By : Jonathan Tompson & Kristofer Schlachter
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#include <iostream>
#include <iterator>

#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/config.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/event.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/memory_object.hpp>
#include <boost/compute/platform.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/user_event.hpp>
#include <boost/compute/version.hpp>
#include <boost/compute/wait_list.hpp>
#include <boost/compute/source.hpp>

#include <boost/thread.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

namespace compute = boost::compute;
namespace po      = boost::program_options;

/***********************Global Stuff***************/

/***********************Kernel Section*****************/

//BOOST_COMPUTE_STRINGIZE_SOURCE helps in reducing the work
//of adding "" to every line. Refer simple_kernel example

// Run visualize_kernel_id example to have better undestatnding
// on global id and local id
const char matrix_multiply_cl[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    //Naive Kernel
    __kernel void matrix_multiply_naive(__global const float *src_a,
                                      __global const float *src_b,
                                      __global float *dest_c,
                                      int width_a,
                                      int height_a,
                                      int width_b,
                                      int height_b)
    {
        int col = get_global_id(0); //Y-axis
        int row = get_global_id(1); //X-axis

        float accumulate_sum = 0;
        for(int k = 0; k < width_a; ++k)
        {
            accumulate_sum += src_a[row * width_a + k ] *
                              src_b[k * width_b + col];
        }
        dest_c[row * width_a + col] = accumulate_sum;
    }
    //Improved kernel
    __kernel void matrix_multiply_block(__global const float *src_a,
                                     __global const float *src_b,
                                     __global  float *src_c,
                                     int width_a,
                                     int height_a,
                                     int width_b,
                                     int height_b)
    {
        int bx = get_group_id(0);
        int by = get_group_id(1);
        int tx = get_local_id(0);
        int ty = get_local_id(1);

        int aBegin  = width_a * BLOCK_SIZE * by;
        int aEnd    = aBegin + width_a - 1;
        int aStep   = BLOCK_SIZE;

        int bBegin  = BLOCK_SIZE * bx;
        int bStep   = BLOCK_SIZE * width_b;

        float Csub = 0.0;

        for(int a = aBegin, b = bBegin; a <= aEnd; a+=aStep,
                                                   b+=bStep)
        {
            //__local specifier makes the variable static
            // to the block
            __local float As[BLOCK_SIZE][BLOCK_SIZE];
            __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

            As[ty][tx] = src_a[a + width_a * ty + tx];
            Bs[ty][tx] = src_b[b + width_b * ty + tx];

            barrier(CLK_LOCAL_MEM_FENCE);

            for(int k = 0; k < BLOCK_SIZE; ++k)
                    Csub += As[ty][k] * Bs[k][tx];

            barrier(CLK_LOCAL_MEM_FENCE);
         }

        int c = width_b * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        src_c[c + width_b * ty + tx] = Csub;
    }
    );

/***********************User Defined Function*****************/

//For ease of understanding same width & height is considered
void matrix_multiply(float* src1, float* src2, float* dest, int dim)
{
    //This is important to understand how we
    //can map to an individual OpenCL thread
    for(int row = 0; row < dim; row++)  //Global ID 1
        for(int col = 0; col < dim; col++) //Global ID 0
        {
            float sum = 0;  //Kernel Private variable
            for(int k = 0; k < dim; k++)    //Thread Loop
                sum += src1[row * dim + k] * src2[k * dim + col];

            dest[row * dim + col] = sum;
        }
}

void fill_matrix(float *data, int size)
{
    for(int i=0; i<size; i++)
    {
        for(int j=0; j<size; j++)
         data[ j * size + i] = 1;
    }
}

void display_matrix(float *data, int size)
{
    for(int i=0; i<size; i++)
    {
        for(int j=0; j<size; j++)
            std::cout<<"  "<<data[ j * size + i];
        std::cout<<std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////
    int user_size;

    po::options_description desc("Allowed Options");
    desc.add_options()
            ("help", "produce help message")
            ("info", "prints the information")
            ("usage", "using example")
            ("size", po::value<int>(),"input size for square matrix")
            ("display", "display the result for user size");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout<<desc<<std::endl;
        return 1;
    }

    if(vm.count("usage"))
    {
        std::cout<<argv[0]<<" --size 1024"<<std::endl;
        std::cout<<argv[0]<<" --size 64 --display"<<std::endl;
        return 1;
    }

    if(vm.count("info"))
    {
        std::cout<<"OpenCL 2D matrix multiplication \n"
                 <<"  *  Using pointers \n"
                 <<"  *  Using naive kernel \n"
                 <<"  *  Using shared memory kernel \n";
        return 1;
    }

    if(vm.count("size"))
    {
        user_size = vm["size"].as<int>();
    }

    ///////////////////////////////////////////////////////////////////////

    std::cout<<"======================================"<<std::endl;
    std::cout<<"Compute Matrix Multiplication example"<<std::endl;
    std::cout<<"======================================"<<std::endl;

    float a[8][8] = { {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1}
                    };
    float b[8][8];
    float c[8][8];

    float *a_ptr;
    float *b_ptr;
    float *c_ptr;

    fill_matrix(&b[0][0], 8);

    ///OpenCL Device Initialization
    //Platform(system) -> Devices -> Context -> Command Queue ->
    //Create Buffers -> Copy data to buffers -> Set Kernel Args -> Run Kernels

    //Profiling says how much time the device has taken to run our kernel

    compute::device  cl_device = compute::system::default_device();
    compute::context cl_device_context(cl_device);
    compute::command_queue cl_device_queue(cl_device_context, cl_device,
                                     compute::command_queue::enable_profiling);
    compute::event profiler;
    uint64_t elapsed;

    ///////////////////////////////////////////////////////////////////////

    std::cout<<"8x8 Matrix Multiplication via pointers"<<std::endl;
    matrix_multiply(&a[0][0], &b[0][0], &c[0][0], 8);
    display_matrix(&c[0][0], 8);

    ///////////////////////////////////////////////////////////////////////

    //Compile program -> Get handle to kernel -> Run kernel
    //Host Buffer -> Device Buffer -> Map data to Kernel ->
    //Run the Kernel -> Copy result back to host pointer

    compute::program matrix_multiply_program =
            compute::program::create_with_source(matrix_multiply_cl,
                                                 cl_device_context);
    try
    {
        matrix_multiply_program.build("-D BLOCK_SIZE=4");
    }
    catch(boost::compute::opencl_error &e)
    {
        std::cout <<"Error : "<<e.what()<<"\n"
                  <<matrix_multiply_program.build_log() << std::endl;
         return -1;
    }

    compute::kernel improved_kernel(matrix_multiply_program,
                                    "matrix_multiply_block");
    compute::kernel naive_kernel(matrix_multiply_program,
                                 "matrix_multiply_naive");

    //Verify everything is going well
    std::cout<<"Kernel Names :  "
             <<naive_kernel.get_info<std::string>(CL_KERNEL_FUNCTION_NAME)
             <<std::endl<<"\t\t"
             <<improved_kernel.get_info<std::string>(CL_KERNEL_FUNCTION_NAME)
             <<std::endl;;

    compute::buffer dev_a(cl_device_context, 8 * 8 * sizeof(float),
                          compute::memory_object::read_only |
                          compute::memory_object::copy_host_ptr,
                          a);

    compute::buffer dev_b(cl_device_context, 8 * 8 * sizeof(float),
                          compute::memory_object::read_only |
                          compute::memory_object::copy_host_ptr,
                          b);

    compute::buffer dev_c(cl_device_context, 8 * 8 * sizeof(float),
                          compute::memory_object::write_only
                          );

    std::cout<<"8x8 Matrix Multiplication via matrix_multiply_block kernel"<<std::endl;
    improved_kernel.set_arg(0, dev_a);
    improved_kernel.set_arg(1, dev_b);
    improved_kernel.set_arg(2, dev_c);
    improved_kernel.set_arg(3, 8);
    improved_kernel.set_arg(4, 8);
    improved_kernel.set_arg(5, 8);
    improved_kernel.set_arg(6, 8);

    size_t global_thread_size[2]  = { 8, 8 };
    size_t local_thread_size[2]   = { 4, 4 };

    cl_device_queue.enqueue_nd_range_kernel( improved_kernel,
                                   2,
                                   0,
                                   global_thread_size,
                                   local_thread_size);
    cl_device_queue.enqueue_read_buffer(dev_c,
                              0,
                              8 * 8 * sizeof(float),
                              c);

    display_matrix(&c[0][0], 8);

    ///////////////////////////////////////////////////////////////////////
    //Gets executed if user inputs the size
    if(vm.count("size"))
    {
        std::cout<<std::endl
                 <<user_size<<" x "<<user_size
                 <<" Matrix Multiplication with both the kernels"
                 <<std::endl;

        a_ptr = (float*)malloc(sizeof(float) * user_size * user_size);
        b_ptr = (float*)malloc(sizeof(float) * user_size * user_size);
        c_ptr = (float*)malloc(sizeof(float) * user_size * user_size);

        fill_matrix(a_ptr, user_size);
        fill_matrix(b_ptr, user_size);
        fill_matrix(c_ptr, user_size);

        std::cout << "Matrix Size: " << user_size << "x"
                  << user_size << std::endl;
        std::cout << "Grid Size  : " << user_size/4 << "x"
                  << user_size/4 << " blocks" << std::endl;
        std::cout << "Local Size : 4 x 4 threads" << std::endl;

        size_t user_global_thread_size[2]  = { user_size, user_size };
        size_t user_local_thread_size[2]   = { 4, 4 };

        try
        {

            compute::buffer dynamic_dev_a(cl_device_context,
                                          user_size * user_size *
                                                    sizeof(float),
                                          compute::memory_object::read_only |
                                          compute::memory_object::copy_host_ptr,
                                          a_ptr);

            compute::buffer dynamic_dev_b(cl_device_context,
                                          user_size * user_size *
                                                    sizeof(float),
                                          compute::memory_object::read_only,
                                          b_ptr);

            //Copying data through queue
            cl_device_queue.enqueue_write_buffer(dynamic_dev_b, 0,
                                                 user_size * user_size *
                                                          sizeof(float),
                                                 b_ptr);

            compute::buffer dynamic_dev_c(cl_device_context,
                                          user_size * user_size *
                                                     sizeof(float),
                                          compute::memory_object::write_only
                                         );

            naive_kernel.set_arg(0, dynamic_dev_a);
            naive_kernel.set_arg(1, dynamic_dev_b);
            naive_kernel.set_arg(2, dynamic_dev_c);
            naive_kernel.set_arg(3, user_size);
            naive_kernel.set_arg(4, user_size);
            naive_kernel.set_arg(5, user_size);
            naive_kernel.set_arg(6, user_size);

            profiler = cl_device_queue.enqueue_nd_range_kernel(
                                                       naive_kernel,
                                                       2,
                                                       0,
                                                       user_global_thread_size,
                                                       user_local_thread_size);
            //Make sure you are waiting for kernel to finish
            cl_device_queue.finish();
            elapsed = profiler.duration<boost::chrono::nanoseconds>().count();

            std::cout << "Time taken with naive kernel   : "
                      << elapsed  << " ns : "
                      << elapsed/100000 << " ms : "
                      << (float)elapsed/1000000000 << " s"
                      << std::endl;

            cl_device_queue.enqueue_read_buffer(dynamic_dev_c,
                                      0,
                                      user_size * user_size * sizeof(float),
                                      c_ptr);

            improved_kernel.set_arg(0, dynamic_dev_a);
            improved_kernel.set_arg(1, dynamic_dev_b);
            improved_kernel.set_arg(2, dynamic_dev_c);
            improved_kernel.set_arg(3, user_size);
            improved_kernel.set_arg(4, user_size);
            improved_kernel.set_arg(5, user_size);
            improved_kernel.set_arg(6, user_size);

            profiler = cl_device_queue.enqueue_nd_range_kernel(
                                                       improved_kernel,
                                                       2,
                                                       0,
                                                       user_global_thread_size,
                                                       user_local_thread_size);

            //Make sure you are waiting for kernel to finish
            cl_device_queue.finish();
            elapsed = profiler.duration<boost::chrono::nanoseconds>().count();

            std::cout << "Time taken with improved kernel: "
                      << elapsed  << " ns : "
                      << elapsed/100000 << " ms : "
                      << (float)elapsed/1000000000 << " s"
                      << std::endl;

            cl_device_queue.enqueue_read_buffer(dynamic_dev_c,
                                      0,
                                      user_size * user_size * sizeof(float),
                                      c_ptr);
        }
        catch(compute::opencl_error e)
        {
            std::cout<<"Error code: "<<e.error_string()<<"\n";
            return -1;
        }

        if(vm.count("display"))
        {
            display_matrix(c_ptr, user_size);
        }
    }
}
