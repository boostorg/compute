//---------------------------------------------------------------------------//
// Copyright (c) 2015 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_MERGE_SORT_ON_CPU_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_MERGE_SORT_ON_CPU_HPP

#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class Iterator, class Compare>
inline void merge_blocks(Iterator first,
                         Iterator result,
                         Compare compare,
                         size_t count,
                         const size_t block_size,
                         command_queue &queue)
{
    meta_kernel k("merge_sort_on_cpu_merge_blocks");
    size_t count_arg = k.add_arg<const uint_>("count");
    size_t block_size_arg = k.add_arg<uint_>("block_size");

    k <<
        k.decl<uint_>("b1_start") << " = get_global_id(0) * block_size * 2;\n" <<
        k.decl<uint_>("b1_end") << " = min(count, b1_start + block_size);\n" <<
        k.decl<uint_>("b2_start") << " = min(count, b1_start + block_size);\n" <<
        k.decl<uint_>("b2_end") << " = min(count, b2_start + block_size);\n" <<
        k.decl<uint_>("result_idx") << " = b1_start;\n" <<

        // merging block 1 and block 2 (stable)
        "while(b1_start < b1_end && b2_start < b2_end){\n" <<
        "    if( " << compare(first[k.var<uint_>("b2_start")],
                              first[k.var<uint_>("b1_start")]) << "){\n" <<
        "        " << result[k.var<uint_>("result_idx")] <<  " = " <<
                      first[k.var<uint_>("b2_start")] << ";\n" <<
        "        b2_start++;\n" <<
        "    }\n" <<
        "    else {\n" <<
        "        " << result[k.var<uint_>("result_idx")] <<  " = " <<
                      first[k.var<uint_>("b1_start")] << ";\n" <<
        "        b1_start++;\n" <<
        "    }\n" <<
        "    result_idx++;\n" <<
        "}\n" <<
        "while(b1_start < b1_end){\n" <<
        "   " << result[k.var<uint_>("result_idx")] <<  " = " <<
                 first[k.var<uint_>("b1_start")] << ";\n" <<
        "    b1_start++;\n" <<
        "    result_idx++;\n" <<
        "}\n" <<
        "while(b2_start < b2_end){\n" <<
        "   " << result[k.var<uint_>("result_idx")] <<  " = " <<
                 first[k.var<uint_>("b2_start")] << ";\n" <<
        "    b2_start++;\n" <<
        "    result_idx++;\n" <<
        "}\n";

    const context &context = queue.get_context();
    ::boost::compute::kernel kernel = k.compile(context);
    kernel.set_arg(count_arg, static_cast<const uint_>(count));
    kernel.set_arg(block_size_arg, static_cast<uint_>(block_size));

    const size_t global_size = static_cast<size_t>(
        std::ceil(float(count) / (2 * block_size))
    );
    queue.enqueue_1d_range_kernel(kernel, 0, global_size, 0);
}

template<class Iterator, class Compare>
inline void block_insertion_sort(Iterator first,
                                 Compare compare,
                                 const size_t count,
                                 const size_t block_size,
                                 command_queue &queue)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    meta_kernel k("merge_sort_on_cpu_block_insertion_sort");
    size_t count_arg = k.add_arg<uint_>("count");
    size_t block_size_arg = k.add_arg<uint_>("block_size");

    k <<
        k.decl<uint_>("start") << " = get_global_id(0) * block_size;\n" <<
        k.decl<uint_>("end") << " = min(count, start + block_size);\n" <<

        // block insertion sort (stable)
        "for(uint i = start+1; i < end; i++){\n" <<
        "    " << k.decl<const T>("value") << " = " << first[k.var<uint_>("i")] << ";\n" <<
        "    uint pos = i;\n" <<
        "    while(pos > start && " <<
                   compare(k.var<const T>("value"),
                           first[k.var<uint_>("pos-1")]) << "){\n" <<
        "        " << first[k.var<uint_>("pos")] << " = " << first[k.var<uint_>("pos-1")] << ";\n" <<
        "        pos--;\n" <<
        "    }\n" <<
        "    " << first[k.var<uint_>("pos")] << " = value;\n" <<
        "}\n"; // block insertion sort

    const context &context = queue.get_context();
    ::boost::compute::kernel kernel = k.compile(context);
    kernel.set_arg(count_arg, static_cast<uint_>(count));
    kernel.set_arg(block_size_arg, static_cast<uint_>(block_size));

    const size_t global_size = static_cast<size_t>(std::ceil(float(count) / block_size));
    queue.enqueue_1d_range_kernel(kernel, 0, global_size, 0);
}

template<class Iterator, class Compare>
inline void merge_sort_on_cpu(Iterator first,
                              Iterator last,
                              Compare compare,
                              command_queue &queue)
{
    typedef typename std::iterator_traits<Iterator>::value_type value_type;

    size_t count = iterator_range_size(first, last);
    if(count < 2){
        return;
    }
    // for small input size only insertion sort is performed
    else if(count <= 512){
        block_insertion_sort(first, compare, count, count, queue);
        return;
    }

    const context &context = queue.get_context();
    const device &device = queue.get_device();

    // loading parameters
    std::string cache_key =
        std::string("__boost_merge_sort_on_cpu_") + type_name<value_type>();
    boost::shared_ptr<parameter_cache> parameters =
        detail::parameter_cache::get_global_cache(device);

    const size_t block_size =
        parameters->get(cache_key, "insertion_sort_block_size", 64);
    block_insertion_sort(first, compare, count, block_size, queue);

    // temporary buffer for merge result
    vector<value_type> temp(count, context);
    bool result_in_temp = false;

    for(size_t i = block_size; i < count; i *= 2){
        result_in_temp = !result_in_temp;
        if(result_in_temp) {
            merge_blocks(first, temp.begin(), compare, count, i, queue);
        } else {
            merge_blocks(temp.begin(), first, compare, count, i, queue);
        }
    }

    // if the result is in temp buffer we need to copy it to input
    if(result_in_temp) {
        copy(temp.begin(), temp.end(), first, queue);
    }
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_MERGE_SORT_ON_CPU_HPP
