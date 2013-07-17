//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_ADJACENT_DIFFERENCE_HPP
#define BOOST_COMPUTE_ALGORITHM_ADJACENT_DIFFERENCE_HPP

#include <iterator>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/iterator/detail/adjacent_transform_iterator.hpp>

namespace boost {
namespace compute {

/// Stores the difference of each pair of consecutive values in the
/// range [\p first, \p last) to the range beginning at \p result.
template<class InputIterator, class OutputIterator>
inline OutputIterator
adjacent_difference(InputIterator first,
                    InputIterator last,
                    OutputIterator result,
                    command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    ::boost::compute::minus<value_type> op;

    return ::boost::compute::copy(
               detail::make_adjacent_transform_iterator(first, op),
               detail::make_adjacent_transform_iterator(last, op),
               result,
               queue
           );
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_ADJACENT_DIFFERENCE_HPP
