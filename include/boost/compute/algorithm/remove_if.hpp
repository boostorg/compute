//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_REMOVE_IF_HPP
#define BOOST_COMPUTE_ALGORITHM_REMOVE_IF_HPP

#include <boost/compute/algorithm/copy_if.hpp>
#include <boost/compute/functional/logical.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class Iterator, class Predicate>
inline Iterator remove_if(Iterator first,
                          Iterator last,
                          Predicate predicate,
                          command_queue &queue)
{
    ::boost::compute::unary_negate<Predicate> not_predicate(predicate);

    return ::boost::compute::copy_if(first,
                                     last,
                                     first,
                                     not_predicate,
                                     queue);
}

template<class Iterator, class Predicate>
inline Iterator remove_if(Iterator first,
                          Iterator last,
                          Predicate predicate)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    return ::boost::compute::remove_if(first, last, predicate, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_REMOVE_IF_HPP
