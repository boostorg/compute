//---------------------------------------------------------------------------//
// Copyright (c) 2015 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_EXTREMA_REDUCE_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_EXTREMA_REDUCE_HPP

#include <algorithm>
#include <vector>

#include <boost/compute/types.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/detail/parameter_cache.hpp>
#include <boost/compute/memory/local_buffer.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/utility/program_cache.hpp>
#include <boost/compute/algorithm/detail/serial_find_extrema.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator>
bool find_extrema_reduce_requirements_met(InputIterator first,
                                          InputIterator last,
                                          command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::value_type input_type;

    const device &device = queue.get_device();

    // device must have dedicated local memory storage
    // otherwise reduction would be highly inefficient
    if(device.get_info<CL_DEVICE_LOCAL_MEM_TYPE>() != CL_LOCAL)
    {
        return false;
    }

    const size_t max_work_group_size = device.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    // local memory size in bytes (per compute unit)
    const size_t local_mem_size = device.get_info<CL_DEVICE_LOCAL_MEM_SIZE>();

    std::string cache_key = std::string("__boost_find_extrema_reduce_")
        + type_name<input_type>();
    // load parameters
    boost::shared_ptr<parameter_cache> parameters =
        detail::parameter_cache::get_global_cache(device);

    // Get preferred work group size
    size_t work_group_size = parameters->get(cache_key, "wgsize", 256);

    work_group_size = (std::min)(max_work_group_size, work_group_size);

    // local memory size needed to perform parallel reduction
    size_t required_local_mem_size = 0;
    // indices size
    required_local_mem_size += sizeof(uint_) * work_group_size;
    // values size
    required_local_mem_size += sizeof(input_type) * work_group_size;

    // at least 4 work groups per compute unit otherwise reduction
    // would be highly inefficient
    return ((required_local_mem_size * 4) <= local_mem_size);
}

template<class InputIterator, class ResultIterator, class Compare>
inline size_t find_extrema_reduce(InputIterator first,
                                  size_t count,
                                  ResultIterator result,
                                  vector<uint_>::iterator result_idx,
                                  size_t work_groups_no,
                                  size_t work_group_size,
                                  Compare compare,
                                  const bool find_minimum,
                                  command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::value_type input_type;

    const context &context = queue.get_context();

    meta_kernel k("find_extrema_reduce");
    size_t count_arg = k.add_arg<uint_>("count");
    size_t output_arg = k.add_arg<input_type *>(memory_object::global_memory, "output");
    size_t output_idx_arg = k.add_arg<uint_ *>(memory_object::global_memory, "output_idx");
    size_t block_arg = k.add_arg<input_type *>(memory_object::local_memory, "block");
    size_t block_idx_arg = k.add_arg<uint_ *>(memory_object::local_memory, "block_idx");

    k <<
        // Work item global id
        k.decl<const uint_>("gid") << " = get_global_id(0);\n" <<
        //
        "if(gid >= count) {\n return;\n }\n" <<

        // Index of element that will be read from input buffer
        k.decl<uint_>("idx") << " = gid;\n" <<

        k.decl<input_type>("acc") << ";\n" <<
        // Index of currently best element
        k.decl<uint_>("acc_idx") << " = idx;\n" <<

        // Init accumulator with first[get_global_id(0)]
        "acc = " << first[k.var<uint_>("idx")] << ";\n" <<
        "idx += get_global_size(0);\n" <<

        k.decl<bool>("compare_result") << ";\n" <<
        "while( idx < count ){\n" <<
            // Next element
            k.decl<input_type>("next") << " = " << first[k.var<uint_>("idx")] << ";\n" <<
            // Comparison between currently best element (acc) and next element
            "#ifndef BOOST_COMPUTE_FIND_MAXIMUM\n" <<
            "compare_result = " << compare(k.var<input_type>("acc"),
                                           k.var<input_type>("next")) << ";\n" <<
            "#else\n" <<
            "compare_result = " << compare(k.var<input_type>("next"),
                                           k.var<input_type>("acc")) << ";\n" <<
            "#endif\n" <<
            "acc = compare_result ? acc : next;\n" <<
            "acc_idx = compare_result ? acc_idx : idx;\n" <<
            "idx += get_global_size(0);\n" <<
        "}\n" <<

        // Work item local id
        k.decl<const uint_>("lid") << " = get_local_id(0);\n" <<
        "block[lid] = acc;\n" <<
        "block_idx[lid] = acc_idx;\n" <<
        "barrier(CLK_LOCAL_MEM_FENCE);\n" <<

        k.decl<uint_>("group_offset") << " = count - (get_local_size(0) * get_group_id(0));\n";

    k <<
        "#pragma unroll\n"
        "for(" << k.decl<uint_>("offset") << " = " << uint_(work_group_size) << " / 2; offset > 0; " <<
             "offset = offset / 2) {\n" <<
             "if((lid < offset) && ((lid + offset) < group_offset)) { \n" <<
                 k.decl<input_type>("mine") << " = block[lid];\n" <<
                 k.decl<input_type>("other") << " = block[lid+offset];\n" <<
                 "#ifndef BOOST_COMPUTE_FIND_MAXIMUM\n" <<
                 "compare_result = " << compare(k.var<input_type>("mine"),
                                                k.var<input_type>("other")) << ";\n" <<
                 "#else\n" <<
                 "compare_result = " << compare(k.var<input_type>("other"),
                                                k.var<input_type>("mine")) << ";\n" <<
                 "#endif\n" <<
                 "block[lid] = compare_result ? mine : other;\n" <<
                 "block_idx[lid] = compare_result ? " <<
                     "block_idx[lid] : block_idx[lid+offset];\n" <<
             "}\n"
             "barrier(CLK_LOCAL_MEM_FENCE);\n" <<
        "}\n" <<

         // write block result to global output
        "if(lid == 0){\n" <<
        "    output[get_group_id(0)] = block[0];\n" <<
        "    output_idx[get_group_id(0)] = block_idx[0];\n" <<
        "}";

    std::string options;
    if(!find_minimum){
        options = "-DBOOST_COMPUTE_FIND_MAXIMUM";
    }
    kernel kernel = k.compile(context, options);

    kernel.set_arg(count_arg, static_cast<uint_>(count));
    kernel.set_arg(output_arg, result.get_buffer());
    kernel.set_arg(output_idx_arg, result_idx.get_buffer());
    kernel.set_arg(block_arg, local_buffer<input_type>(work_group_size));
    kernel.set_arg(block_idx_arg, local_buffer<uint_>(work_group_size));

    queue.enqueue_1d_range_kernel(kernel,
                                  0,
                                  work_groups_no * work_group_size,
                                  work_group_size);

    return 0;
}

template<class InputIterator, class Compare>
uint_ find_extrema_final(InputIterator candidates,
                         vector<uint_>::iterator candidates_idx,
                         const size_t count,
                         Compare compare,
                         const bool find_minimum,
                         const size_t work_group_size,
                         command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::value_type input_type;

    const context &context = queue.get_context();

    // device vectors for the result
    vector<input_type> result(1, context);
    vector<uint_> result_idx(1, context);

    // get extremum from among the candidates
    find_extrema_reduce(
        candidates, count, result.begin(), result_idx.begin(),
        1, work_group_size, compare, find_minimum, queue
    );

    // get candidate index
    const uint_ idx = (result_idx.begin()).read(queue);
    // get extremum index
    typename vector<uint_>::iterator extremum_idx = candidates_idx + idx;

    // return extremum index
    return extremum_idx.read(queue);
}

template<class InputIterator>
uint_ find_extrema_final(InputIterator candidates,
                         vector<uint_>::iterator candidates_idx,
                         const size_t count,
                         ::boost::compute::less<
                             typename std::iterator_traits<InputIterator>::value_type
                         > compare,
                         const bool find_minimum,
                         const size_t work_group_size,
                         command_queue &queue)
{
    (void) work_group_size;

    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;
    typedef typename std::iterator_traits<InputIterator>::value_type input_type;

    // host vectors
    std::vector<input_type> host_candidates(count);
    std::vector<uint_> host_candidates_idx(count);

    InputIterator candidates_last =
        candidates + static_cast<difference_type>(count);
    vector<uint_>::iterator candidates_idx_last =
        candidates_idx + count;

    // copying extremum candidates found by find_extrema_reduce(...) to host
    ::boost::compute::copy(candidates_idx, candidates_idx_last,
                           host_candidates_idx.begin(), queue);
    ::boost::compute::copy(candidates, candidates_last,
                           host_candidates.begin(), queue);

    typename std::vector<input_type>::iterator i = host_candidates.begin();
    std::vector<uint_>::iterator idx = host_candidates_idx.begin();
    std::vector<uint_>::iterator extremum_idx = idx;
    input_type extremum = *i;

    // find extremum from among the candidates
    if(!find_minimum) {
        while(idx != host_candidates_idx.end()) {
            bool compare_result =  *i > extremum;
            extremum = compare_result ? *i : extremum;
            extremum_idx = compare_result ? idx : extremum_idx;
            idx++, i++;
        }
    }
    else {
        while(idx != host_candidates_idx.end()) {
            bool compare_result =  *i < extremum;
            extremum = compare_result ? *i : extremum;
            extremum_idx = compare_result ? idx : extremum_idx;
            idx++, i++;
        }
    }

    // return extremum index
    return (*extremum_idx);
}

template<class InputIterator, class Compare>
InputIterator find_extrema_reduce(InputIterator first,
                                  InputIterator last,
                                  Compare compare,
                                  const bool find_minimum,
                                  command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;
    typedef typename std::iterator_traits<InputIterator>::value_type input_type;

    const context &context = queue.get_context();
    const device &device = queue.get_device();

    // Getting information about used queue and device
    const size_t compute_units_no = device.get_info<CL_DEVICE_MAX_COMPUTE_UNITS>();
    const size_t max_work_group_size = device.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    const size_t count = detail::iterator_range_size(first, last);

    std::string cache_key = std::string("__boost_find_extrema_reduce_")
        + type_name<input_type>();

    // load parameters
    boost::shared_ptr<parameter_cache> parameters =
        detail::parameter_cache::get_global_cache(device);

    // get preferred work group size and preferred number
    // of work groups per compute unit
    size_t work_group_size = parameters->get(cache_key, "wgsize", 256);
    size_t work_groups_per_cu = parameters->get(cache_key, "wgpcu", 64);

    // calculate work group size and number of work groups
    work_group_size = (std::min)(max_work_group_size, work_group_size);
    size_t work_groups_no = compute_units_no * work_groups_per_cu;
    work_groups_no = (std::min)(
            work_groups_no,
            static_cast<size_t>(std::ceil(float(count) / work_group_size)));

    // device vectors for extremum candidates and their indices
    vector<input_type> candidates(work_groups_no, context);
    vector<uint_> candidates_idx(work_groups_no, context);

    // find extremum candidates and their indices
    find_extrema_reduce(
        first, count, candidates.begin(), candidates_idx.begin(),
        work_groups_no, work_group_size, compare, find_minimum, queue
     );

    // get extremum index
    const uint_ extremum_idx = find_extrema_final(
        candidates.begin(), candidates_idx.begin(), work_groups_no, compare,
        find_minimum, work_group_size, queue
    );

    return first + static_cast<difference_type>(extremum_idx);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_EXTREMA_REDUCE_HPP
