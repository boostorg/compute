//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_UPPER_BOUND_HPP
#define BOOST_COMPUTE_ALGORITHM_UPPER_BOUND_HPP

#include <boost/compute/lambda.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/find_if.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class InputIterator, class T>
inline InputIterator upper_bound(InputIterator first,
                                 InputIterator last,
                                 const T &value,
                                 command_queue &queue)
{
    using ::boost::compute::_1;

    InputIterator position =
        ::boost::compute::find_if(first, last, _1 > value, queue);

    return position;
}

template<class InputIterator, class T>
inline InputIterator upper_bound(InputIterator first,
                                 InputIterator last,
                                 const T &value)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::upper_bound(first, last, value, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_UPPER_BOUND_HPP
