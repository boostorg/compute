//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_PARTITION_HPP
#define BOOST_COMPUTE_ALGORITHM_PARTITION_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy_if.hpp>
#include <boost/compute/container/vector.hpp>

namespace boost {
namespace compute {

template<class Iterator, class UnaryPredicate>
inline Iterator partition(Iterator first,
                          Iterator last,
                          UnaryPredicate predicate,
                          command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<Iterator>::value_type value_type;

    const context &context = queue.get_context();

    // make temporary copy of the input
    ::boost::compute::vector<value_type> tmp(first, last, context);

    // copy true values
    Iterator last_true =
        ::boost::compute::copy_if(tmp.begin(),
                                  tmp.end(),
                                  first,
                                  predicate,
                                  queue);

    // copy false values
    Iterator last_false =
        ::boost::compute::copy_if(tmp.begin(),
                                  tmp.end(),
                                  last_true,
                                  unary_negate<UnaryPredicate>(predicate),
                                  queue);

    // return iterator pointing to the last true value
    return last_true;
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_PARTITION_HPP
