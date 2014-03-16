//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_NTH_ELEMENT_HPP
#define BOOST_COMPUTE_ALGORITHM_NTH_ELEMENT_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/sort.hpp>

namespace boost {
namespace compute {

/// Rearranges the elements in the range [\p first, \p last) such that
/// the \p nth element would be in that position in a sorted sequence.
template<class Iterator, class Compare>
inline void nth_element(Iterator first,
                        Iterator nth,
                        Iterator last,
                        Compare compare,
                        command_queue &queue = system::default_queue())
{
    (void) nth;

    sort(first, last, compare, queue);
}

/// \overload
template<class Iterator>
inline void nth_element(Iterator first,
                        Iterator nth,
                        Iterator last,
                        command_queue &queue = system::default_queue())
{
    (void) nth;

    sort(first, last, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_NTH_ELEMENT_HPP
