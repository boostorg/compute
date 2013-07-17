//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_ADJACENT_FIND_HPP
#define BOOST_COMPUTE_ALGORITHM_ADJACENT_FIND_HPP

#include <iterator>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/iterator/detail/adjacent_transform_iterator.hpp>

namespace boost {
namespace compute {

/// Searches the range [\p first, \p last) for two identical adjacent
/// elements and returns an iterator pointing to the first.
template<class InputIterator>
inline InputIterator
adjacent_find(InputIterator first,
              InputIterator last,
              command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    ::boost::compute::equal_to<value_type> eq;

    return ::boost::compute::find(
               detail::make_adjacent_transform_iterator(first, eq),
               detail::make_adjacent_transform_iterator(last, eq),
               true,
               queue
           ).base() - 1;
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_ADJACENT_FIND_HPP
