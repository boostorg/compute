//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_IF_WITH_ATOMICS_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_IF_WITH_ATOMICS_HPP

#include <iterator>

#include <boost/compute/types.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/detail/scalar.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class UnaryPredicate>
inline InputIterator find_if_with_atomics(InputIterator first,
                                          InputIterator last,
                                          UnaryPredicate predicate,
                                          command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;
    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;

    size_t count = detail::iterator_range_size(first, last);
    if(count == 0){
        return last;
    }

    const context &context = queue.get_context();

    detail::meta_kernel k("find_if");
    size_t index_arg = k.add_arg<int *>(memory_object::global_memory, "index");
    atomic_min<uint_> atomic_min_uint;

    k << k.decl<const uint_>("i") << " = get_global_id(0);\n"
      << k.decl<const value_type>("value") << "="
      <<     first[k.var<const uint_>("i")] << ";\n"
      << "if(" << predicate(k.var<const value_type>("value")) << "){\n"
      << "    " << atomic_min_uint(k.var<uint_ *>("index"), k.var<uint_>("i")) << ";\n"
      << "}\n";

    kernel kernel = k.compile(context);

    scalar<uint_> index(context);
    kernel.set_arg(index_arg, index.get_buffer());

    // initialize index to the last iterator's index
    index.write(static_cast<uint_>(count), queue);

    queue.enqueue_1d_range_kernel(kernel, 0, count, 0);

    // read index and return iterator
    return first + static_cast<difference_type>(index.read(queue));
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_IF_WITH_ATOMICS_HPP
