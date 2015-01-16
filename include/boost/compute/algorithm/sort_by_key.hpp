//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_SORT_BY_KEY_HPP
#define BOOST_COMPUTE_ALGORITHM_SORT_BY_KEY_HPP

#include <iterator>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/detail/insertion_sort.hpp>
#include <boost/compute/algorithm/detail/radix_sort.hpp>
#include <boost/compute/algorithm/reverse.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {

namespace detail {

template<class KeyIterator, class ValueIterator>
inline void dispatch_sort_by_key(KeyIterator keys_first,
                        KeyIterator keys_last,
                        ValueIterator values_first,
                        less<typename std::iterator_traits<KeyIterator>::value_type> compare,
                        command_queue &queue)
{
    size_t count = detail::iterator_range_size(keys_first, keys_last);

    if(count < 32){
        detail::serial_insertion_sort_by_key(
            keys_first, keys_last, values_first, compare, queue
            );
    }
    else {
        detail::radix_sort_by_key(
            keys_first, keys_last, values_first, queue
            );
    }
}

template<class KeyIterator, class ValueIterator>
inline void dispatch_sort_by_key(KeyIterator keys_first,
                        KeyIterator keys_last,
                        ValueIterator values_first,
                        greater<typename std::iterator_traits<KeyIterator>::value_type> compare,
                        command_queue &queue)
{
    size_t count = detail::iterator_range_size(keys_first, keys_last);

    if(count < 32){
        detail::serial_insertion_sort_by_key(
            keys_first, keys_last, values_first, compare, queue
            );
    }
    else {
        // radix sorts in ascending order
        detail::radix_sort_by_key(
            keys_first, keys_last, values_first, queue
            );

        // Reverse keys, values for descending order
        ::boost::compute::reverse(keys_first, keys_last, queue);
        ::boost::compute::reverse(values_first, values_first + count, queue);

    }
}

template<class KeyIterator, class ValueIterator, class Compare>
inline void dispatch_sort_by_key(KeyIterator keys_first,
                        KeyIterator keys_last,
                        ValueIterator values_first,
                        Compare compare,
                        command_queue &queue)
{

    detail::serial_insertion_sort_by_key(
        keys_first,
        keys_last,
        values_first,
        compare,
        queue
        );
}

}

/// Performs a key-value sort using the keys in the range [\p keys_first,
/// \p keys_last) on the values in the range [\p values_first,
/// \p values_first \c + (\p keys_last \c - \p keys_first)) using \p compare.
///
/// If no compare function is specified, \c less is used.
///
/// \see sort()

template<class KeyIterator, class ValueIterator, class Compare>
inline void sort_by_key(KeyIterator keys_first,
                        KeyIterator keys_last,
                        ValueIterator values_first,
                        Compare compare,
                        command_queue &queue = system::default_queue())
{
    ::boost::compute::detail::dispatch_sort_by_key(
        keys_first,
        keys_last,
        values_first,
        compare,
        queue);
}

/// \overload
template<class KeyIterator, class ValueIterator>
inline void sort_by_key(KeyIterator keys_first,
                        KeyIterator keys_last,
                        ValueIterator values_first,
                        command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<KeyIterator>::value_type key_type;

    size_t count = detail::iterator_range_size(keys_first, keys_last);

    if(count < 32){
        detail::serial_insertion_sort_by_key(
            keys_first, keys_last, values_first, less<key_type>(), queue
        );
    }
    else {
        detail::radix_sort_by_key(
            keys_first, keys_last, values_first, queue
        );
    }
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_SORT_BY_KEY_HPP
