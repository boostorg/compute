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
#include <boost/compute/iterator/transform_iterator.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/functional/detail/unpack.hpp>

namespace boost {
namespace compute {

/// Returns \c true if the values in the range [\p first, \p last)
/// are in sorted order.
template<class InputIterator>
inline bool is_sorted(InputIterator first,
                      InputIterator last,
                      command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    ::boost::compute::less_equal<value_type> op;

    return boost::get<0>(
        ::boost::compute::find(
            ::boost::compute::make_transform_iterator(
                ::boost::compute::make_zip_iterator(
                    boost::make_tuple(first, first + 1)
                ),
                detail::unpack(op)
            ),
            ::boost::compute::make_transform_iterator(
                ::boost::compute::make_zip_iterator(
                    boost::make_tuple(last - 1, last)
                ),
                detail::unpack(op)
            ),
            false,
            queue
        ).base().get_iterator_tuple()
    ) == last - 1;
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_IS_SORTED_HPP
