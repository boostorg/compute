//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_MAX_ELEMENT_HPP
#define BOOST_COMPUTE_ALGORITHM_MAX_ELEMENT_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/detail/find_extrema.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class InputIterator>
inline InputIterator max_element(InputIterator first,
                                 InputIterator last,
                                 command_queue &queue)
{
    return detail::find_extrema(first, last, '>', queue);
}

template<class InputIterator>
inline InputIterator max_element(InputIterator first,
                                 InputIterator last)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::max_element(first, last, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_MAX_ELEMENT_HPP
