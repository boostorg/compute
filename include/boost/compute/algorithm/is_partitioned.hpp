//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_IS_PARTITIONED_HPP
#define BOOST_COMPUTE_ALGORITHM_IS_PARTITIONED_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/find_if.hpp>
#include <boost/compute/algorithm/find_if_not.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class InputIterator, class UnaryPredicate>
inline bool is_partitioned(InputIterator first,
                           InputIterator last,
                           UnaryPredicate predicate,
                           command_queue &queue)
{
    return ::boost::compute::find_if(
               ::boost::compute::find_if_not(first,
                                             last,
                                             predicate,
                                             queue),
                last,
                predicate,
                queue) == last;
}

template<class InputIterator, class UnaryPredicate>
inline bool is_partitioned(InputIterator first,
                           InputIterator last,
                           UnaryPredicate predicate)
{
    command_queue &queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::is_partitioned(first, last, predicate, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_PARTITION_HPP
