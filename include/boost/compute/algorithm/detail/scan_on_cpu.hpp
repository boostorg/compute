//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_SCAN_ON_CPU_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_SCAN_ON_CPU_HPP

#include <iterator>

#include <boost/compute/device.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class OutputIterator>
inline OutputIterator scan_on_cpu(InputIterator first,
                                  InputIterator last,
                                  OutputIterator result,
                                  bool exclusive,
                                  command_queue &queue)
{
    if(first == last){
        return result;
    }

    typedef typename
        std::iterator_traits<InputIterator>::value_type input_type;

    const context &context = queue.get_context();

    // create scan kernel
    meta_kernel k("scan_on_cpu");
    k.add_arg<ulong_>("n");
    k <<
        k.decl<input_type>("sum") << " = 0;\n" <<
        "for(ulong i = 0; i < n; i++){\n" <<
        k.decl<const input_type>("x") << " = "
            << first[k.var<ulong_>("i")] << ";\n";

    if(exclusive){
        k << result[k.var<ulong_>("i")] << " = sum;\n";
    }

    k << "    sum = sum + x;\n";

    if(!exclusive){
        k << result[k.var<ulong_>("i")] << " = sum;\n";
    }

    k << "}\n";

    // compile scan kernel
    kernel scan_kernel = k.compile(context);

    // setup kernel arguments
    size_t n = detail::iterator_range_size(first, last);
    scan_kernel.set_arg<ulong_>(0, n);

    // execute the kernel
    queue.enqueue_1d_range_kernel(scan_kernel, 0, 1, 1);

    // return iterator pointing to the end of the result range
    return result + n;
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_SCAN_ON_CPU_HPP
