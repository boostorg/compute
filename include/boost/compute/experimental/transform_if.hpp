//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_EXPERIMENTAL_TRANSFORM_IF_HPP
#define BOOST_COMPUTE_EXPERIMENTAL_TRANSFORM_IF_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/algorithm/detail/copy_on_device.hpp>

namespace boost {
namespace compute {
namespace experimental {

template<class InputIterator,
         class OutputIterator,
         class UnaryOperator,
         class Predicate>
inline OutputIterator transform_if(InputIterator first,
                                   InputIterator last,
                                   OutputIterator result,
                                   UnaryOperator op,
                                   Predicate predicate,
                                   command_queue &queue)
{
    typedef typename
        std::iterator_traits<InputIterator>::difference_type difference_type;

    difference_type count = std::distance(first, last);
    if(count < 1){
        return result;
    }

    detail::meta_kernel k("transform_if");

    k <<
        k.if_(predicate(first[k.get_global_id(0)])) << "{\n" <<
            result[k.get_global_id(0)] << '=' <<
                op(first[k.get_global_id(0)]) << ";\n"
        "}\n";

    const device &device = queue.get_device();
    const size_t work_group_size =
        detail::pick_copy_work_group_size(count, device);

    k.exec_1d(queue, 0, count, work_group_size);

    return result + count;
}

} // end experimental namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_TRANSFORM_IF_HPP
