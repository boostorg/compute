//---------------------------------------------------------------------------//
// Copyright (c) 2014 Roshan <thisisroshansmail@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_UNIQUE_COPY_HPP
#define BOOST_COMPUTE_ALGORITHM_UNIQUE_COPY_HPP

#include <boost/compute/algorithm/copy_if.hpp>
#include <boost/compute/algorithm/gather.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/iterator/detail/adjacent_transform_iterator.hpp>
#include <boost/compute/lambda.hpp>
#include <boost/compute/system.hpp>

namespace boost {
namespace compute {

/// Makes a copy of the range [first, last) and removes
/// all consecutive duplicate elements from the copy 
/// Returns an iterator for the new logical end of the range.
template<class InputIterator, class OutputIterator>
inline OutputIterator unique_copy(InputIterator first, 
                                  InputIterator last,
                                  OutputIterator result,
                                  command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;
    size_t count = detail::iterator_range_size(first, last);

    equal_to<value_type> eq;

    vector<uint_> temp(count, queue.get_context());

    vector<uint_>::iterator iter = copy_index_if(
            detail::make_adjacent_transform_iterator(first, eq),
            detail::make_adjacent_transform_iterator(last, eq),
            temp.begin(),
            _1 != true,
            queue
        );

    gather(temp.begin(), iter, first, result, queue);

    return result + detail::iterator_range_size(temp.begin(), iter);
}

}
}

#endif // BOOST_COMPUTE_ALGORITHM_UNIQUE_COPY_HPP
