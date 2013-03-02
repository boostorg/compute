//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_HISTOGRAM_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_HISTOGRAM_HPP

#include <iterator>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class OutputIterator>
class histogram_kernel : public meta_kernel
{
public:
    histogram_kernel(InputIterator first,
                     OutputIterator result)
        : meta_kernel("histogram")
    {
        typedef typename std::iterator_traits<InputIterator>::value_type value_type;

        *this <<
            "atomic_inc(&" << result[first[get_global_id(0)]] << ");\n";
    }
};

template<class InputIterator, class OutputIterator>
inline void histogram(InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      command_queue &queue)
{
    size_t count = iterator_range_size(first, last);
    if(count == 0){
        return;
    }

    histogram_kernel<InputIterator, OutputIterator> kernel(first, result);
    kernel.exec_1d(queue, 0, count);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_HISTOGRAM_HPP
