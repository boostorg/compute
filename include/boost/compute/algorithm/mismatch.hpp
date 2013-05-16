//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_MISMATCH_HPP
#define BOOST_COMPUTE_ALGORITHM_MISMATCH_HPP

#include <iterator>
#include <utility>

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/iterator/detail/binary_transform_iterator.hpp>

namespace boost {
namespace compute {

template<class InputIterator1, class InputIterator2>
inline std::pair<InputIterator1, InputIterator2>
mismatch(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2,
         command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator1>::value_type value_type;

    ::boost::compute::equal_to<value_type> op;

    InputIterator2 last2 = first2 + std::distance(first1, last1);

    InputIterator1 iter =
        ::boost::compute::find(
            detail::make_binary_transform_iterator(first1, first2, op),
            detail::make_binary_transform_iterator(last1, last2, op),
            false,
            queue
        ).base();

    return std::make_pair(iter, first2 + std::distance(first1, iter));
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_MISMATCH_HPP
