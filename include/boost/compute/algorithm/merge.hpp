//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_MERGE_HPP
#define BOOST_COMPUTE_ALGORITHM_MERGE_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/detail/merge_with_merge_path.hpp>
#include <boost/compute/algorithm/detail/serial_merge.hpp>

namespace boost {
namespace compute {

/// Merges the sorted values in the range [\p first1, \p last1) with the sorted
/// values in the range [\p first2, last2) and stores the result in the range
/// beginning at \p result. Values are compared using the \p comp function. If
/// no comparision function is given, \c less is used.
///
/// \param first1 first element in the first range to merge
/// \param last1 last element in the first range to merge
/// \param first2 first element in the second range to merge
/// \param last2 last element in the second range to merge
/// \param result first element in the result range
/// \param comp comparison function (by default \c less)
/// \param queue command queue to perform the operation
///
/// \return \c OutputIterator to the end of the result range
///
/// \see inplace_merge()
template<class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class Compare>
inline OutputIterator merge(InputIterator1 first1,
                            InputIterator1 last1,
                            InputIterator2 first2,
                            InputIterator2 last2,
                            OutputIterator result,
                            Compare comp,
                            command_queue &queue = system::default_queue())
{
    return detail::merge_with_merge_path(first1, last1, first2, last2, result, comp, queue);
}

/// \overload
template<class InputIterator1, class InputIterator2, class OutputIterator>
inline OutputIterator merge(InputIterator1 first1,
                            InputIterator1 last1,
                            InputIterator2 first2,
                            InputIterator2 last2,
                            OutputIterator result,
                            command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator1>::value_type value_type;
    less<value_type> less_than;
    return merge(first1, last1, first2, last2, result, less_than, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_MERGE_HPP
