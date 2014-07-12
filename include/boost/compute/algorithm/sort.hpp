//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_SORT_HPP
#define BOOST_COMPUTE_ALGORITHM_SORT_HPP

#include <iterator>

#include <boost/utility/enable_if.hpp>

#include <boost/compute/buffer.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/detail/fixed_sort.hpp>
#include <boost/compute/algorithm/detail/radix_sort.hpp>
#include <boost/compute/algorithm/detail/insertion_sort.hpp>
#include <boost/compute/container/mapped_view.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

// sort() for device iterators
template <class Iterator>
inline void dispatch_sort(Iterator first,
                          Iterator last,
                          command_queue &queue,
                          typename boost::enable_if<
                              is_device_iterator<Iterator>
                          >::type* = 0)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    size_t count = detail::iterator_range_size(first, last);
    if(count < 2){
        return;
    }
    else if(count == 2){
        ::boost::compute::detail::sort2<T>(first.get_buffer(), queue);
    }
    else if(count == 3){
        ::boost::compute::detail::sort3<T>(first.get_buffer(), queue);
    }
    else if(count <= 32){
        ::boost::compute::detail::serial_insertion_sort(first, last, queue);
    }
    else {
        ::boost::compute::detail::radix_sort(first, last, queue);
    }
}

// sort() for host iterators
template <class Iterator>
inline void dispatch_sort(Iterator first,
                          Iterator last,
                          command_queue &queue,
                          typename boost::disable_if<
                              is_device_iterator<Iterator>
                          >::type* = 0)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    size_t size = static_cast<size_t>(std::distance(first, last));

    // create mapped buffer
    mapped_view<T> view(
        boost::addressof(*first), size, queue.get_context()
    );

    // sort mapped buffer
    dispatch_sort(view.begin(), view.end(), queue);

    // return results to host
    view.map(queue);
}

} // end detail namespace

/// Sorts the values in the range [\p first, \p last) according to
/// \p compare.
///
/// \param first first element in the range to sort
/// \param last last element in the range to sort
/// \param compare comparison function (by default \c less)
/// \param queue command queue to perform the operation
///
/// For example, to sort a vector on the device:
/// \code
/// // create vector on the device with data
/// float data[] = { 2.f, 4.f, 1.f, 3.f };
/// boost::compute::vector<float> vec(data, data + 4, queue);
///
/// // sort the vector on the device
/// boost::compute::sort(vec.begin(), vec.end(), queue);
/// \endcode
///
/// The sort() algorithm can also be directly used with host iterators. This
/// example will automatically transfer the data to the device, sort it, and
/// then transfer the data back to the host:
/// \code
/// std::vector<int> data = { 9, 3, 2, 5, 1, 4, 6, 7 };
///
/// boost::compute::sort(data.begin(), data.end(), queue);
/// \endcode
///
/// \see is_sorted()
template<class Iterator, class Compare>
inline void sort(Iterator first,
                 Iterator last,
                 Compare compare,
                 command_queue &queue = system::default_queue())
{
    size_t count = detail::iterator_range_size(first, last);
    if(count < 2){
        return;
    }

    return ::boost::compute::detail::serial_insertion_sort(first,
                                                           last,
                                                           compare,
                                                           queue);
}

/// \overload
template<class Iterator>
inline void sort(Iterator first,
                 Iterator last,
                 command_queue &queue = system::default_queue())
{
    detail::dispatch_sort(first, last, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_SORT_HPP
