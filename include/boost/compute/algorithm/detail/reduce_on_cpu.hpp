//---------------------------------------------------------------------------//
// Copyright (c) 2015 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_SERIAL_REDUCE_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_SERIAL_REDUCE_HPP

#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/type_traits/result_of.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class OutputIterator, class BinaryFunction>
inline void reduce_on_cpu(InputIterator first,
                          InputIterator last,
                          OutputIterator result,
                          BinaryFunction function,
                          command_queue &queue)
{
    typedef typename
        std::iterator_traits<InputIterator>::value_type T;
    typedef typename
        ::boost::compute::result_of<BinaryFunction(T, T)>::type result_type;

    const context &context = queue.get_context();
    size_t count = detail::iterator_range_size(first, last);
    if(count == 0){
        return;
    }

    meta_kernel k("reduce_on_cpu");
    const size_t compute_units = queue.get_device().compute_units();
    buffer output(context, sizeof(result_type) * compute_units);

    size_t count_arg = k.add_arg<uint_>("count");
    size_t output_arg =
        k.add_arg<result_type *>(memory_object::global_memory, "output");

    k <<
        "uint block = " <<
            "(uint)ceil(((float)count)/get_global_size(0));\n" <<
        "uint index = get_global_id(0) * block;\n" <<
        "uint end = min(count, index + block);\n" <<

        k.decl<result_type>("result") << " = " << first[k.var<uint_>("index")] << ";\n" <<
        "index++;\n" <<
        "while(index < end){\n" <<
             "result = " << function(k.var<T>("result"),
                                     first[k.var<uint_>("index")]) << ";\n" <<
             "index++;\n" <<
        "}\n" <<
        "output[get_global_id(0)] = result;\n";

    size_t global_work_size = compute_units;
    if(count <= 1024) global_work_size = 1;
    kernel kernel = k.compile(context);

    if(global_work_size == 1) {
        kernel.set_arg(count_arg, static_cast<uint_>(count));
        kernel.set_arg(output_arg, result.get_buffer());
        queue.enqueue_1d_range_kernel(kernel, 0, global_work_size, 0);
        return;
    }

    kernel.set_arg(count_arg, static_cast<uint_>(count));
    kernel.set_arg(output_arg, output);
    queue.enqueue_1d_range_kernel(kernel, 0, global_work_size, 0);

    reduce_on_cpu(
        make_buffer_iterator<result_type>(output),
        make_buffer_iterator<result_type>(output, global_work_size),
        result,
        function,
        queue
    );
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_SERIAL_REDUCE_HPP
