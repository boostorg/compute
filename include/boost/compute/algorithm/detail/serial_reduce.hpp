//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_SERIAL_REDUCE_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_SERIAL_REDUCE_HPP

#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class T, class BinaryFunction>
inline T serial_reduce(InputIterator first,
                       InputIterator last,
                       T init,
                       BinaryFunction function,
                       command_queue &queue)
{
    size_t count = detail::iterator_range_size(first, last);
    if(count == 0){
        return init;
    }

    const context &context = queue.get_context();

    meta_kernel k("serial_reduce");
    size_t init_arg = k.add_arg<T>("init");
    size_t count_arg = k.add_arg<cl_uint>("count");
    size_t output_arg = k.add_arg<T *>("__global", "output");

    k <<
        k.decl<T>("result") << " = init;\n" <<
        "for(uint i = 0; i < count; i++)\n" <<
        "    result = " << function(k.var<T>("result"),
                                    first[k.var<cl_uint>("i")]) << ";\n" <<
        "*output = result;\n";

    kernel kernel = k.compile(context);

    buffer output_buffer(context, sizeof(T));

    kernel.set_arg(init_arg, init);
    kernel.set_arg(count_arg, static_cast<cl_uint>(count));
    kernel.set_arg(output_arg, output_buffer);

    queue.enqueue_task(kernel);

    return detail::read_single_value<T>(output_buffer, queue);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_SERIAL_REDUCE_HPP
