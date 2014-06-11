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

const char matrix_multiply_naive_cl[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void mat_multiply_naive(__global const float *src_a,
                                      __global const float *src_b,
                                      __global float *dest_c,
                                      int width_a, int height_a,
                                      int width_b, int height_b)
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
    );

// Run visualize_kernel example to have better undestatnding on blocks
const char matrix_multiply_block_cl[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void mat_multiply_block(__global const float *src_a,
                                     __global const float *src_b,
                                     __global  float *src_c,
                                     int width_a, int height_a,
                                     int width_b, int height_b)
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

void display_matrix(float *data, int dim)
{
    for(int i=0; i<dim; i++)
    {
        for(int j=0; j<dim; j++)
            std::cout<<"  "<<data[ j * dim + i];
        std::cout<<std::endl;
    }
}


/***********************Thread Functions*****************/
int test_matrix_multiply_array(compute::context& context,
                                         compute::command_queue& queue)
{
    float a[4][4] = { {1, 1, 1, 1},
                      {1, 1, 1, 1},
                      {1, 1, 1, 1},
                      {1, 1, 1, 1},
                    };
    float b[4][4] = { {1, 1, 1, 1},
                      {1, 1, 1, 1},
                      {1, 1, 1, 1},
                      {1, 1, 1, 1},
                    };
    float c[4][4];
    float accumulate_sum = 0;

    ///////////////////////////////////////////////////////////////////////
    std::cout<<"Local 4x4 Matrix Multiplication with known index"<<std::endl;
    {
        for(int row=0; row<4; row++)
            for(int col=0; col<4; col++)
            {
                accumulate_sum = 0;
                for(int k=0; k<4; k++)
                    accumulate_sum += a[row][k] * b[k][col];
                c[row][col] = accumulate_sum;
            }

        for(int i=0; i<4; i++)
        {
            for(int j=0; j<4; j++)
                std::cout<<"  "<<c[i][j];
            std::cout<<std::endl;
        }
    }

    ///////////////////////////////////////////////////////////////////////
    std::cout<<"4x4 Matrix Multiplication via pointers"<<std::endl;
    matrix_multiply(&a[0][0], &b[0][0], &c[0][0], 4);
    display_matrix(&c[0][0], 4);

    ///////////////////////////////////////////////////////////////////////
    std::cout<<"4X4 Matrix Multiplication via naive kernel"<<std::endl;

    /// Compile program -> Get handle to kernel -> Run kernel

    /// Host Buffer -> Device Buffer -> Map data to Kernel ->
    /// Run the Kernel -> Copy result back to host pointer


    //Create program from char array
    compute::program matrix_multiply_program =
            compute::program::create_with_source(matrix_multiply_naive_cl,
                                                 context);

    //Build the OpenCL code on the provided vendor compiler
    try
    {
           matrix_multiply_program.build();
    }
    catch(boost::compute::opencl_error &e)
    {
          std::cout <<"OpenCL Build Error : \n"
                   << matrix_multiply_program.build_log() << std::endl;
         return -1;
    }

    compute::kernel naive_kernel(matrix_multiply_program,
                                 "mat_multiply_naive");
    ///! TODO: Replicate Kernel set args error

    std::cout<<"Kernel Name : "
             <<naive_kernel.get_info<std::string>(CL_KERNEL_FUNCTION_NAME)
             <<std::endl;

    //Create buffer on device RAM and make sure you copy data to it.
    //Which can be done with CL_MEM_COPY_HOST_PTR ~ copy_host_ptr
    compute::buffer dev_a(context, 4 * 4 * sizeof(float),
                          compute::memory_object::read_only |
                          compute::memory_object::copy_host_ptr,
                          a);

    compute::buffer dev_b(context, 4 * 4 * sizeof(float),
                          compute::memory_object::read_only, b);
    //Copying data through queue
    queue.enqueue_write_buffer(dev_b, 0, 4 * 4 * sizeof(int), b);

    compute::buffer dev_c(context, 4 * 4 * sizeof(float));

    //Map the data to respective kernel args
    naive_kernel.set_arg(0, dev_a);
    naive_kernel.set_arg(1, dev_b);
    naive_kernel.set_arg(2, dev_c);
    naive_kernel.set_arg(3, 4);
    naive_kernel.set_arg(4, 4);
    naive_kernel.set_arg(5, 4);
    naive_kernel.set_arg(6, 4);

    //Global thread size in X by Y
    const size_t global_thread_size[2] = {4,4};

    //Run the kernel
    queue.enqueue_nd_range_kernel(naive_kernel,
                                  2,
                                  0,
                                  global_thread_size,
                                  0);

    //Copy result to host pointer &c[0][0](base address of 2D array)
    queue.enqueue_read_buffer(dev_c,
                              0,
                              4 * 4 * sizeof(float),
                              &c[0][0]);

    display_matrix(&c[0][0], 4);
}

int test_block_matrix_multiply(compute::context& context,
                                compute::command_queue& queue)
{
    float a[8][8] = { {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1}
                    };
    float b[8][8] = { {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1, 1, 1, 1}
                    };
    float c[8][8];

    ///////////////////////////////////////////////////////////////////////
    std::cout<<"8x8 Matrix Multiplication via pointers"<<std::endl;
    matrix_multiply(&a[0][0], &b[0][0], &c[0][0], 8);
    display_matrix(&c[0][0], 8);

    ///////////////////////////////////////////////////////////////////////
    std::cout<<"8x8 Matrix Multiplication via improved kernel"<<std::endl;
    std::string options;
    std::string buildLogs;

    compute::program matrix_block_multiply_program =
            compute::program::create_with_source(matrix_multiply_block_cl,
                                                 context);
    try
    {
        matrix_block_multiply_program.build("-D BLOCK_SIZE=4");
    }
    catch(boost::compute::opencl_error &e)
    {
        std::cout <<"OpenCL Build Error : \n"
                  <<matrix_block_multiply_program.build_log() << std::endl;
         return -1;
    }

    compute::kernel improved_kernel(matrix_block_multiply_program,
                                    "mat_multiply_block");

    //Verify everything is going well
    std::cout<<"Kernel Name : "
             <<improved_kernel.get_info<std::string>(CL_KERNEL_FUNCTION_NAME)
             <<std::endl;

    compute::buffer dev_a(context, 8 * 8 * sizeof(float),
                          compute::memory_object::read_only |
                          compute::memory_object::copy_host_ptr,
                          a);

    compute::buffer dev_b(context, 8 * 8 * sizeof(float),
                          compute::memory_object::read_only |
                          compute::memory_object::copy_host_ptr,
                          b);

    compute::buffer dev_c(context, 8 * 8 * sizeof(float),
                          compute::memory_object::write_only
                          );

    improved_kernel.set_arg(0, dev_a);
    improved_kernel.set_arg(1, dev_b);
    improved_kernel.set_arg(2, dev_c);
    improved_kernel.set_arg(3, 8);
    improved_kernel.set_arg(4, 8);
    improved_kernel.set_arg(5, 8);
    improved_kernel.set_arg(6, 8);

    size_t global_thread_size[2]  = { 8, 8 };
    size_t local_thread_size[2]   = { 4, 4 };

    queue.enqueue_nd_range_kernel( improved_kernel,
                                   2,
                                   0,
                                   global_thread_size,
                                   local_thread_size);
    queue.enqueue_read_buffer(dev_c,
                              0,
                              8 * 8 * sizeof(float),
                              c);

    display_matrix(&c[0][0], 8);

}

//"ReadMe:"
//    You will learn OpenCL matrix multiplication and how to
//    run OpenCL application in multi threads
//    *  2D Matrix Multiplication using i,j,k
//    *  2D Matrix Multiplication using pointers
//    *  2D Matrix Multiplication using naive kernel
//    *  2D Matrix Multiplication using shared memory kernel
//    *  Using context and queue in multiple threads

int main(int argc, char *argv[])
{
    std::cout<<"============================"<<std::endl;
    std::cout<<"Compute Multi Thread example"<<std::endl;
    std::cout<<"============================"<<std::endl;

    po::options_description desc("Allowed Options");
    desc.add_options()
            ("help", "produce help message")
            ("thread_use", po::value<int>(), "1(yes)/0(no)")
            ("usage", "how to run");

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
        std::cout<<argv[0]<<" --thread_use=1 \n";
        return 1;
    }

    if(vm.count("thread_use"))
    {
        std::cout<<"Boost Thread set to : "
                 <<vm["thread_use"].as<int>()
                 <<std::endl;
    }
    else
    {
        std::cout<<"To get help: "<<argv[0]<<" --help"
                   <<std::endl;
        return 1;
    }

    //OpenCL Device Initialization
    //Platform(system) -> Devices -> Context -> Command Queue ->
    //Create Buffers -> Copy data to buffers -> Set Kernel Args -> Run Kernels

    //Profiling says how much time the device has taken to run our kernel

    compute::device  gpu = compute::system::default_device();
    compute::context gpu_context(gpu);
    compute::command_queue gpu_queue(gpu_context, gpu,
                                     compute::command_queue::enable_profiling);

    if(vm["thread_use"].as<int>())
    {
        //Create a thread variables with the function address,
        //that its need to run
        boost::thread simple_array_multiply_thread(&test_matrix_multiply_array,
                                                   gpu_context,
                                                   gpu_queue);

        boost::thread array_block_multiply_thread(&test_block_matrix_multiply,
                                                   gpu_context,
                                                   gpu_queue);

        //Wait for the threads to complete
        simple_array_multiply_thread.join();
        array_block_multiply_thread.join();

        std::cout<<std::endl<<"!!!IMPORTANT!!!:"
                 <<" With multithread enabled you see unorganized prints"
                 <<std::endl
                 <<std::endl;
    }
    else
    {
        test_matrix_multiply_array(gpu_context, gpu_queue);
        test_block_matrix_multiply(gpu_context, gpu_queue);
    }
}

// !TODO: Finding GFLOPS
// Try Different height and width
