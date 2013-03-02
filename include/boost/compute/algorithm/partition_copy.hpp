//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_PARTITION_COPY_HPP
#define BOOST_COMPUTE_ALGORITHM_PARTITION_COPY_HPP

#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy_if.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class InputIterator,
         class OutputIterator1,
         class OutputIterator2,
         class UnaryPredicate>
inline std::pair<OutputIterator1, OutputIterator2>
partition_copy(InputIterator first,
               InputIterator last,
               OutputIterator1 first_true,
               OutputIterator2 first_false,
               UnaryPredicate predicate,
               command_queue &queue)
{
    // copy true values
    OutputIterator1 last_true =
        ::boost::compute::copy_if(first,
                                  last,
                                  first_true,
                                  predicate,
                                  queue);

    // copy false values
    OutputIterator2 last_false =
        ::boost::compute::copy_if(first,
                                  last,
                                  first_false,
                                  unary_negate<UnaryPredicate>(predicate),
                                  queue);

    // return iterators to the end of the true and the false ranges
    return std::make_pair(last_true, last_false);
}

template<class InputIterator,
         class OutputIterator1,
         class OutputIterator2,
         class UnaryPredicate>
inline std::pair<OutputIterator1, OutputIterator2>
partition_copy(InputIterator first,
               InputIterator last,
               OutputIterator1 first_true,
               OutputIterator2 first_false,
               UnaryPredicate predicate)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::partition_copy(first,
                                            last,
                                            first_true,
                                            first_false,
                                            predicate,
                                            queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_PARTITION_COPY_HPP
