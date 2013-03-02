//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_SCATTER_HPP
#define BOOST_COMPUTE_ALGORITHM_SCATTER_HPP

#include <boost/algorithm/string/replace.hpp>

#include <boost/compute/exception.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputBufferType, class MapBufferType, class OutputBufferType>
struct scatter_kernel
{
    static std::string source()
    {
        std::string source =
            "__kernel void scatter(__global const $T1 *input,\n"
            "                      const uint input_offset,\n"
            "                      __global const $T2 *map,\n"
            "                      __global $T3 *output,\n"
            "                      const uint output_offset)\n"
            "{\n"
            "    const uint i = get_global_id(0);\n"
            "    output[map[i] + output_offset] = input[i + input_offset];\n"
            "}\n";

        boost::replace_all(source, "$T1", type_name<InputBufferType>());
        boost::replace_all(source, "$T2", type_name<MapBufferType>());
        boost::replace_all(source, "$T3", type_name<OutputBufferType>());

        return source;
    }
};

} // end detail namespace

template<class InputIterator, class MapIterator, class OutputIterator>
inline void scatter(InputIterator first,
                    InputIterator last,
                    MapIterator map,
                    OutputIterator result,
                    command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::value_type input_value_type;
    typedef typename std::iterator_traits<MapIterator>::value_type map_value_type;
    typedef typename std::iterator_traits<OutputIterator>::value_type output_value_type;

    size_t count = detail::iterator_range_size(first, last);
    if(count == 0){
        // nothing to do
        return;
    }

    const context &context = queue.get_context();
    std::string source =
        detail::scatter_kernel<input_value_type,
                               map_value_type,
                               output_value_type>::source();
    kernel kernel = kernel::create_with_source(source, "scatter", context);

    kernel.set_arg(0, first.get_buffer());
    kernel.set_arg(1, static_cast<cl_uint>(first.get_index()));
    kernel.set_arg(2, map.get_buffer());
    kernel.set_arg(3, result.get_buffer());
    kernel.set_arg(4, static_cast<cl_uint>(result.get_index()));

    queue.enqueue_1d_range_kernel(kernel, 0, count);
}

template<class InputIterator, class MapIterator, class OutputIterator>
inline void scatter(InputIterator first,
                    InputIterator last,
                    MapIterator map,
                    OutputIterator result)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    ::boost::compute::scatter(first, last, map, result, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_SCATTER_HPP
