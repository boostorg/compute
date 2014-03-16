//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_REVERSE_COPY_HPP
#define BOOST_COMPUTE_ALGORITHM_REVERSE_COPY_HPP

#include <iterator>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/reverse.hpp>

namespace boost {
namespace compute {

/// Copies the elements in the range [\p first, \p last) in reversed
/// order to the range beginning at \p result.
///
/// \see reverse()
template<class InputIterator, class OutputIterator>
inline OutputIterator
reverse_copy(InputIterator first,
             InputIterator last,
             OutputIterator result,
             command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<OutputIterator>::difference_type difference_type;

    difference_type count = std::distance(first, last);

    // copy data to result
    ::boost::compute::copy(first, last, result, queue);

    // reverse result
    ::boost::compute::reverse(result, result + count, queue);

    // return iterator to the end of result
    return result + count;
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_REVERSE_COPY_HPP
