//---------------------------------------------------------------------------//
// Copyright (c) 2014 Roshan <thisisroshansmail@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_UNIQUE_HPP
#define BOOST_COMPUTE_ALGORITHM_UNIQUE_HPP

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/unique_copy.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/system.hpp>

namespace boost {
namespace compute {

/// Removes all consecutive duplicate elements from the range [first, last) 
/// Returns an iterator for the new logical end of the range.
template<class InputIterator>
inline InputIterator unique(InputIterator first, 
                            InputIterator last,
                            command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;
    size_t count = detail::iterator_range_size(first, last);

    vector<value_type> temp(count, queue.get_context());

    buffer_iterator<value_type> iter = unique_copy(first, last, temp.begin(), queue);

    copy(temp.begin(), iter, first, queue);

    return first + detail::iterator_range_size(temp.begin(), iter);
}

}
}

#endif // BOOST_COMPUTE_ALGORITHM_UNIQUE_HPP
