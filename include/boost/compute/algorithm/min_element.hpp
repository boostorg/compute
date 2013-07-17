//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_MIN_ELEMENT_HPP
#define BOOST_COMPUTE_ALGORITHM_MIN_ELEMENT_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/detail/find_extrema.hpp>

namespace boost {
namespace compute {

/// Returns an iterator pointing to the element in range
/// [\p first, \p last) with the minumum value.
///
/// \see max_element()
template<class InputIterator>
inline InputIterator
min_element(InputIterator first,
            InputIterator last,
            command_queue &queue = system::default_queue())
{
    return detail::find_extrema(first, last, '<', queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_MIN_ELEMENT_HPP
