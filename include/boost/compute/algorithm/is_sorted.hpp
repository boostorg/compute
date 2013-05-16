//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_IS_SORTED_HPP
#define BOOST_COMPUTE_ALGORITHM_IS_SORTED_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/iterator/detail/binary_transform_iterator.hpp>

namespace boost {
namespace compute {

template<class InputIterator>
inline bool is_sorted(InputIterator first,
                      InputIterator last,
                      command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    return ::boost::compute::find(
               detail::make_binary_transform_iterator(
                   first,
                   first + 1,
                   ::boost::compute::less_equal<value_type>()
               ),
               detail::make_binary_transform_iterator(
                   last - 1,
                   last,
                   ::boost::compute::less_equal<value_type>()
               ),
               false,
               queue
           ).base() == last - 1;
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_IS_SORTED_HPP
