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

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/algorithm/detail/serial_reduce.hpp>
#include <boost/compute/container/vector.hpp>

namespace boost {
namespace compute {

/// Returns the sum of the elements in the range [\p first, \p last)
/// plus \p init.
template<class InputIterator, class T>
inline T accumulate(InputIterator first,
                    InputIterator last,
                    T init,
                    command_queue &queue = system::default_queue())
{
    T result;
    ::boost::compute::reduce(first, last, &result, init, plus<T>(), queue);
    return result;
}

/// Returns the result of applying \p function to the elements in the
/// range [\p first, \p last) and \p init.
template<class InputIterator, class T, class BinaryFunction>
inline T accumulate(InputIterator first,
                    InputIterator last,
                    T init,
                    BinaryFunction function,
                    command_queue &queue = system::default_queue())
{
    vector<T> result_value(1, queue.get_context());
    detail::serial_reduce(first, last, result_value.begin(), init, function, queue);

    T result;
    ::boost::compute::copy_n(result_value.begin(), 1, &result, queue);
    return result;
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_ACCUMULATE_HPP
