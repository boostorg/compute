//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_INCLUSIVE_SCAN_HPP
#define BOOST_COMPUTE_ALGORITHM_INCLUSIVE_SCAN_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/detail/scan.hpp>

namespace boost {
namespace compute {

/// Performs an inclusive scan on the elements in the range
/// [\p first, \p last) and stores the results in the range
/// beginning at \p result.
///
/// \see exclusive_scan()
template<class InputIterator, class OutputIterator>
inline OutputIterator
inclusive_scan(InputIterator first,
               InputIterator last,
               OutputIterator result,
               command_queue &queue = system::default_queue())
{
    return detail::scan(first, last, result, false, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_INCLUSIVE_SCAN_HPP
