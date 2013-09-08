//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_RADIX_SORT_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_RADIX_SORT_HPP

#include <iterator>

#include <boost/assert.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_floating_point.hpp>

#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/detail/program_cache.hpp>

namespace boost {
namespace compute {
namespace detail {

template<size_t N>
struct radix_sort_value_type
{
};

template<>
struct radix_sort_value_type<1>
{
    typedef uchar_ type;
};

template<>
struct radix_sort_value_type<2>
{
    typedef ushort_ type;
};

template<>
struct radix_sort_value_type<4>
{
    typedef uint_ type;
};

template<>
struct radix_sort_value_type<8>
{
    typedef ulong_ type;
};

const char radix_sort_source[] =
"#define K2 (1 << K)\n"
"#define RADIX_MASK ((((T)(1)) << K) - 1)\n"
"#define SIGN_BIT ((sizeof(T) * CHAR_BIT) - 1)\n"

"inline uint radix(const T x, const uint low_bit)\n"
"{\n"
"#if defined(IS_FLOATING_POINT)\n"
"    const T mask = -(x >> SIGN_BIT) | (((T)(1)) << SIGN_BIT);\n"
"    return ((x ^ mask) >> low_bit) & RADIX_MASK;\n"
"#elif defined(IS_SIGNED)\n"
"    return ((x ^ (((T)(1)) << SIGN_BIT)) >> low_bit) & RADIX_MASK;\n"
"#else\n"
"    return (x >> low_bit) & RADIX_MASK;\n"
"#endif\n"
"}\n"

"__kernel void count(__global const T *input,\n"
"                    const uint input_size,\n"
"                    __global uint *global_counts,\n"
"                    __global uint *global_offsets,\n"
"                    __local uint *local_counts,\n"
"                    const uint low_bit)\n"
"{\n"
     // work-item parameters
"    const uint gid = get_global_id(0);\n"
"    const uint lid = get_local_id(0);\n"

     // zero local counts
"    if(lid < K2){\n"
"        local_counts[lid] = 0;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"

     // reduce local counts
"    if(gid < input_size){\n"
"        T value = input[gid];\n"
"        uint bucket = radix(value, low_bit);\n"
"        atomic_inc(local_counts + bucket);\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"

     // write block-relative offsets
"    if(lid < K2){\n"
"        global_counts[K2*get_group_id(0) + lid] = local_counts[lid];\n"

         // write global offsets
"        if(get_group_id(0) == (get_num_groups(0) - 1)){\n"
"            global_offsets[lid] = local_counts[lid];\n"
"        }\n"
"    }\n"
"}\n"

"__kernel void scan(__global const uint *block_offsets,\n"
"                   __global uint *global_offsets,\n"
"                   const uint block_count)\n"
"{\n"
"    __global const uint *last_block_offsets =\n"
"        block_offsets + K2 * (block_count - 1);\n"

     // calculate and scan global_offsets
"    uint sum = 0;\n"
"    for(uint i = 0; i < K2; i++){\n"
"        uint x = global_offsets[i] + last_block_offsets[i];\n"
"        global_offsets[i] = sum;\n"
"        sum += x;\n"
"    }\n"
"}\n"

"__kernel void scatter(__global const T *input,\n"
"                      const uint input_size,\n"
"                      const uint low_bit,\n"
"                      __global const uint *counts,\n"
"                      __global const uint *global_offsets,\n"
"                      __global T *output)\n"
"{\n"
     // work-item parameters
"    const uint gid = get_global_id(0);\n"
"    const uint lid = get_local_id(0);\n"

     // copy input to local memory
"    T value;\n"
"    uint bucket;\n"
"    __local uint local_input[BLOCK_SIZE];\n"
"    if(gid < input_size){\n"
"        value = input[gid];\n"
"        bucket = radix(value, low_bit);\n"
"        local_input[lid] = bucket;\n"
"    }\n"

     // copy block counts to local memory
"    __local uint local_counts[(1 << K)];\n"
"    if(lid < K2){\n"
"        local_counts[lid] = counts[get_group_id(0) * K2 + lid];\n"
"    }\n"

     // wait until local memory is ready
"    barrier(CLK_LOCAL_MEM_FENCE);\n"

"    if(gid >= input_size){\n"
"        return;\n"
"    }\n"

     // get global offset
"    uint offset = global_offsets[bucket] + local_counts[bucket];\n"

     // calculate local offset
"    uint local_offset = 0;\n"
"    for(uint i = 0; i < lid; i++){\n"
"        if(local_input[i] == bucket)\n"
"            local_offset++;\n"
"    }\n"

     // write value to output
"    output[offset + local_offset] = value;\n"
"}\n";

template<class Iterator>
inline void radix_sort(Iterator first,
                       Iterator last,
                       command_queue &queue)
{
    typedef typename
        std::iterator_traits<Iterator>::value_type
        value_type;
    typedef typename
        radix_sort_value_type<sizeof(value_type)>::type
        sort_type;

    const context &context = queue.get_context();

    boost::shared_ptr<program_cache> cache =
        detail::get_program_cache(context);

    size_t count = detail::iterator_range_size(first, last);

    // sort parameters
    const uint_ k = 4;
    const uint_ k2 = 1 << k;
    const uint_ block_size = 128;

    uint_ block_count = static_cast<uint_>(count / block_size);
    if(block_count * block_size != count){
        block_count++;
    }

    // load (or create) radix sort program
    std::string cache_key =
        std::string("radix_sort_") + type_name<value_type>();

    program radix_sort_program = cache->get(cache_key);

    if(!radix_sort_program.get()){
        std::stringstream options;
        options << "-DK=" << k;
        options << " -DT=" << type_name<sort_type>();
        options << " -DBLOCK_SIZE=" << block_size;

        if(boost::is_floating_point<value_type>::value){
            options << " -DIS_FLOATING_POINT";
        }

        if(boost::is_signed<value_type>::value){
            options << " -DIS_SIGNED";
        }

        radix_sort_program =
            program::create_with_source(radix_sort_source, context);

        radix_sort_program.build(options.str());

        cache->insert(cache_key, radix_sort_program);
    }

    kernel count_kernel(radix_sort_program, "count");
    kernel scan_kernel(radix_sort_program, "scan");
    kernel scatter_kernel(radix_sort_program, "scatter");

    // setup temporary buffers
    vector<value_type> output(count, context);
    vector<uint_> offsets(k2, context);
    vector<uint_> counts(block_count * k2, context);

    const buffer *input_buffer = &first.get_buffer();
    const buffer *output_buffer = &output.get_buffer();

    for(uint_ i = 0; i < sizeof(sort_type) * CHAR_BIT / k; i++){
        // write counts
        count_kernel.set_arg(0, *input_buffer);
        count_kernel.set_arg(1, static_cast<uint_>(count));
        count_kernel.set_arg(2, counts.get_buffer());
        count_kernel.set_arg(3, offsets.get_buffer());
        count_kernel.set_arg(4, block_size * sizeof(uint_), 0);
        count_kernel.set_arg(5, i * k);
        queue.enqueue_1d_range_kernel(count_kernel,
                                      0,
                                      block_count * block_size,
                                      block_size);

        // scan counts
        if(k == 1){
            typedef uint2_ counter_type;
            ::boost::compute::exclusive_scan(
                make_buffer_iterator<counter_type>(counts.get_buffer(), 0),
                make_buffer_iterator<counter_type>(counts.get_buffer(), counts.size() / 2),
                make_buffer_iterator<counter_type>(counts.get_buffer()),
                queue
            );
        }
        else if(k == 2){
            typedef uint4_ counter_type;
            ::boost::compute::exclusive_scan(
                make_buffer_iterator<counter_type>(counts.get_buffer(), 0),
                make_buffer_iterator<counter_type>(counts.get_buffer(), counts.size() / 4),
                make_buffer_iterator<counter_type>(counts.get_buffer()),
                queue
            );
        }
        else if(k == 4){
            typedef uint16_ counter_type;
            ::boost::compute::exclusive_scan(
                make_buffer_iterator<counter_type>(counts.get_buffer(), 0),
                make_buffer_iterator<counter_type>(counts.get_buffer(), counts.size() / 16),
                make_buffer_iterator<counter_type>(counts.get_buffer()),
                queue
            );
        }
        else {
            BOOST_ASSERT(false && "unknown k");
            break;
        }

        // scan global offsets
        scan_kernel.set_arg(0, counts.get_buffer());
        scan_kernel.set_arg(1, offsets.get_buffer());
        scan_kernel.set_arg(2, block_count);
        queue.enqueue_task(scan_kernel);

        // scatter values
        scatter_kernel.set_arg(0, *input_buffer);
        scatter_kernel.set_arg(1, static_cast<uint_>(count));
        scatter_kernel.set_arg(2, i * k);
        scatter_kernel.set_arg(3, counts.get_buffer());
        scatter_kernel.set_arg(4, offsets.get_buffer());
        scatter_kernel.set_arg(5, *output_buffer);
        queue.enqueue_1d_range_kernel(scatter_kernel,
                                      0,
                                      block_count * block_size,
                                      block_size);

        // swap buffers
        std::swap(input_buffer, output_buffer);
    }
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_RADIX_SORT_HPP
