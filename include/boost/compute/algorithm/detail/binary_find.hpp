//---------------------------------------------------------------------------//
// Copyright (c) 2014 Roshan <thisisroshansmail@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_BINARY_FIND_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_BINARY_FIND_HPP

#include <boost/compute/algorithm/find_if.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/command_queue.hpp>

namespace boost {
namespace compute {
namespace detail{

///
/// \brief Binary find algorithm
///
/// Finds the end of true values in the partitioned range [first, last).
/// \return Iterator pointing to end of true values
///
/// \param first Iterator pointing to start of range
/// \param last Iterator pointing to end of range
/// \param value Value to be found
/// \param queue Queue on which to execute
///
template<class InputIterator, class UnaryPredicate>
inline InputIterator binary_find(InputIterator first,
                                 InputIterator last,
                                 UnaryPredicate predicate,
                                 command_queue &queue = system::default_queue())
{
    return find_if(first, last, predicate, queue);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_BINARY_FIND_HPP
