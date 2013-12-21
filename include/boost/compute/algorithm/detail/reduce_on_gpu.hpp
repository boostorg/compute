//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_REDUCE_ON_GPU_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_REDUCE_ON_GPU_HPP

#include <iterator>

#include <boost/compute/program.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/program_cache.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class T>
inline void reduce_on_gpu(const buffer_iterator<T> first,
                          const buffer_iterator<T> last,
                          const buffer_iterator<T> result,
                          command_queue &queue)
{
    const char source[] = BOOST_STRINGIZE(
        __kernel void reduce(__global const T *input,
                             const uint size,
                             __global T *output)
        {
            __global const T *block = input + get_group_id(0) * VPT * TPB;

            const uint gid = get_global_id(0);
            const uint lid = get_local_id(0);

            __local T scratch[TPB];

            // private reduction
            T sum = 0;
            for(uint i = 0; i < VPT; i++){
                if(block + lid + i*TPB < input + size){
                    sum += block[lid+i*TPB];
                }
            }

            scratch[lid] = sum;

            // local reduction
            for(int i = 1; i < TPB; i <<= 1){
                barrier(CLK_LOCAL_MEM_FENCE);
                uint mask = (i << 1) - 1;
                if((lid & mask) == 0){
                    scratch[lid] += scratch[lid+i];
                }
            }

            // write sum to output
            if(lid == 0){
                output[get_group_id(0)] = scratch[0];
            }
        }
    );

    uint_ vpt = 8;
    uint_ tpb = 128;

    size_t count = std::distance(first, last);

    const context &context = queue.get_context();
    boost::shared_ptr<program_cache> cache = get_program_cache(context);
    std::string cache_key = std::string("boost_reduce_on_gpu_") + type_name<T>();
    program reduce_program = cache->get(cache_key);
    if(!reduce_program.get()){
        // create reduce program
        std::stringstream options;
        options << "-DT=" << type_name<T>()
                << " -DVPT=" << vpt
                << " -DTPB=" << tpb;
        reduce_program = program::create_with_source(source, context);
        reduce_program.build(options.str());

        cache->insert(cache_key, reduce_program);
    }

    // create reduce kernel
    kernel reduce_kernel(reduce_program, "reduce");

    // first pass, reduce from input to ping
    buffer ping(context, std::ceil(float(count) / vpt / tpb) * sizeof(T));

    reduce_kernel.set_arg(0, first.get_buffer());
    reduce_kernel.set_arg(1, uint_(count));
    reduce_kernel.set_arg(2, ping);

    size_t work_size = std::ceil(float(count) / vpt);
    if(work_size % tpb != 0){
        work_size += tpb - work_size % tpb;
    }

    queue.enqueue_1d_range_kernel(reduce_kernel, 0, work_size, tpb);

    count = std::ceil(float(count) / vpt / tpb);

    // middle pass(es), reduce between ping and pong
    const buffer *input_buffer = &ping;
    buffer pong(context, count / vpt / tpb * sizeof(T));
    const buffer *output_buffer = &pong;
    if(count > vpt * tpb){
        while(count > vpt * tpb){
            reduce_kernel.set_arg(0, *input_buffer);
            reduce_kernel.set_arg(1, uint_(count));
            reduce_kernel.set_arg(2, *output_buffer);

            work_size = std::ceil(float(count) / vpt);
            if(work_size % tpb != 0){
                work_size += tpb - work_size % tpb;
            }
            queue.enqueue_1d_range_kernel(reduce_kernel, 0, work_size, tpb);

            std::swap(input_buffer, output_buffer);
            count = std::ceil(float(count) / vpt / tpb);
        }
    }

    // final pass, reduce from ping/pong to result
    reduce_kernel.set_arg(0, *input_buffer);
    reduce_kernel.set_arg(1, uint_(count));
    reduce_kernel.set_arg(2, result.get_buffer());

    queue.enqueue_1d_range_kernel(reduce_kernel, 0, tpb, tpb);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_REDUCE_ON_GPU_HPP
