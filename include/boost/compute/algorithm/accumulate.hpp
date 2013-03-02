//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_ACCUMULATE_HPP
#define BOOST_COMPUTE_ALGORITHM_ACCUMULATE_HPP

#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>
#include <boost/compute/algorithm/detail/serial_reduce.hpp>

namespace boost {
namespace compute {

template<class InputIterator, class T>
inline T accumulate(InputIterator first,
                    InputIterator last,
                    T init,
                    command_queue &queue)
{
    return ::boost::compute::reduce(first, last, init, plus<T>(), queue);
}

template<class InputIterator, class T>
inline T accumulate(InputIterator first,
                    InputIterator last,
                    T init)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::accumulate(first, last, init, queue);
}

template<class InputIterator, class T, class BinaryFunction>
inline T accumulate(InputIterator first,
                    InputIterator last,
                    T init,
                    BinaryFunction function,
                    command_queue &queue)
{
    return detail::serial_reduce(first, last, init, function, queue);
}

template<class InputIterator, class T, class BinaryFunction>
inline T accumulate(InputIterator first,
                    InputIterator last,
                    T init,
                    BinaryFunction function)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::accumulate(first, last, init, function, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_ACCUMULATE_HPP
