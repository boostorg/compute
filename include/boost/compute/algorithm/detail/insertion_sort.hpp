//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_INSERTION_SORT_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_INSERTION_SORT_HPP

#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>


namespace boost {
namespace compute {
namespace detail {

template<class Iterator, class Compare>
inline void serial_insertion_sort(Iterator first,
                                  Iterator last,
                                  Compare compare,
                                  command_queue &queue)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    size_t count = iterator_range_size(first, last);
    if(count < 2){
        return;
    }

    meta_kernel k("serial_insertion_sort");
    size_t local_data_arg = k.add_arg<T *>("__local", "data");
    size_t count_arg = k.add_arg<uint_>("n");

    ::boost::compute::less<T> op;

    k <<
        // copy data to local memory
        "for(uint i = 0; i < n; i++){\n" <<
        "    data[i] = " << first[k.var<uint_>("i")] << ";\n"
        "}\n"

        // sort data in local memory
        "for(uint i = 1; i < n; i++){\n" <<
        "    " << k.decl<const T>("value") << " = data[i];\n" <<
        "    uint pos = i;\n" <<
        "    while(pos > 0 && " <<
                   compare(k.var<const T>("value"),
                           k.var<const T>("data[pos-1]")) << "){\n" <<
        "        data[pos] = data[pos-1];\n" <<
        "        pos--;\n" <<
        "    }\n" <<
        "    data[pos] = value;\n" <<
        "}\n" <<

        // copy sorted data to output
        "for(uint i = 0; i < n; i++){\n" <<
        "    " << first[k.var<uint_>("i")] << " = data[i];\n"
        "}\n";

    const context &context = queue.get_context();
    ::boost::compute::kernel kernel = k.compile(context);
    kernel.set_arg(local_data_arg, static_cast<uint_>(count), 0);
    kernel.set_arg(count_arg, static_cast<uint_>(count));

    queue.enqueue_task(kernel);
}

template<class Iterator>
inline void serial_insertion_sort(Iterator first,
                                  Iterator last,
                                  command_queue &queue)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    ::boost::compute::less<T> less;

    return serial_insertion_sort(first, last, less, queue);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_INSERTION_SORT_HPP
