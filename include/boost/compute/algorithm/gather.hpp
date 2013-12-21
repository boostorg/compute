//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_GATHER_HPP
#define BOOST_COMPUTE_ALGORITHM_GATHER_HPP

#include <boost/algorithm/string/replace.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputBufferType, class MapBufferType, class OutputBufferType>
struct gather_kernel
{
    static std::string source()
    {
        std::string source =
            "__kernel void gather(__global const $T1 *input,\n"
            "                      __global const $T2 *map,\n"
            "                      __global $T3 *output)\n"
            "{\n"
            "    const uint i = get_global_id(0);\n"
            "    output[i] = input[map[i]];\n"
            "}\n";

        boost::replace_all(source, "$T1", type_name<InputBufferType>());
        boost::replace_all(source, "$T2", type_name<MapBufferType>());
        boost::replace_all(source, "$T3", type_name<OutputBufferType>());

        return source;
    }
};

} // end detail namespace

/// Copies the elements using the indices from the range [\p first, \p last)
/// to the range beginning at \p result using the input values from the range
/// beginning at \p input.
///
/// \see scatter()
template<class InputIterator, class MapIterator, class OutputIterator>
inline void gather(MapIterator first,
                   MapIterator last,
                   InputIterator input,
                   OutputIterator result,
                   command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type input_value_type;
    typedef typename std::iterator_traits<MapIterator>::value_type map_value_type;
    typedef typename std::iterator_traits<OutputIterator>::value_type output_value_type;

    const context &context = queue.get_context();
    std::string source =
        detail::gather_kernel<input_value_type,
                               map_value_type,
                               output_value_type>::source();
    kernel kernel = kernel::create_with_source(source, "gather", context);

    kernel.set_arg(0, input.get_buffer());
    kernel.set_arg(1, first.get_buffer());
    kernel.set_arg(2, result.get_buffer());

    size_t offset = first.get_index();
    size_t count = detail::iterator_range_size(first, last);
    queue.enqueue_1d_range_kernel(kernel, offset, count, 0);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_GATHER_HPP
