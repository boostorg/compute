//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_TO_HOST_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_TO_HOST_HPP

#include <iterator>

#include <boost/utility/addressof.hpp>

#include <boost/compute/future.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class DeviceIterator, class HostIterator>
inline HostIterator copy_to_host(DeviceIterator first,
                                 DeviceIterator last,
                                 HostIterator result,
                                 command_queue &queue)
{
    typedef typename
        std::iterator_traits<DeviceIterator>::value_type
        value_type;
    typedef typename
        std::iterator_traits<HostIterator>::difference_type
        difference_type;

    size_t count = iterator_range_size(first, last);
    if(count == 0){
        return result;
    }

    const buffer &buffer = first.get_buffer();
    size_t offset = first.get_index();

    queue.enqueue_read_buffer(buffer,
                              offset * sizeof(value_type),
                              count * sizeof(value_type),
                              ::boost::addressof(*result));

    return result + static_cast<difference_type>(count);
}

template<class DeviceIterator, class HostIterator>
inline future<HostIterator> copy_to_host_async(DeviceIterator first,
                                               DeviceIterator last,
                                               HostIterator result,
                                               command_queue &queue)
{
    typedef typename
        std::iterator_traits<DeviceIterator>::value_type
        value_type;
    typedef typename
        std::iterator_traits<HostIterator>::difference_type
        difference_type;

    size_t count = iterator_range_size(first, last);
    if(count == 0){
        return future<HostIterator>();
    }

    const buffer &buffer = first.get_buffer();
    size_t offset = first.get_index();

    event event_ =
        queue.enqueue_read_buffer_async(buffer,
                                        offset * sizeof(value_type),
                                        count * sizeof(value_type),
                                        ::boost::addressof(*result));

    return make_future(result + static_cast<difference_type>(count), event_);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_TO_HOST_HPP
