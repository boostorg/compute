//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_FILL_HPP
#define BOOST_COMPUTE_ALGORITHM_FILL_HPP

#include <iterator>

#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/future.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/iterator/constant_iterator.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

namespace mpl = boost::mpl;

// fills the range [first, first + count) with value using copy()
template<class BufferIterator, class T>
inline void fill_with_copy(BufferIterator first,
                           size_t count,
                           const T &value,
                           command_queue &queue)
{
    ::boost::compute::copy(
        ::boost::compute::make_constant_iterator(value, 0),
        ::boost::compute::make_constant_iterator(value, count),
        first,
        queue
    );
}

// fills the range [first, first + count) with value using copy_async()
template<class BufferIterator, class T>
inline future<void> fill_async_with_copy(BufferIterator first,
                                         size_t count,
                                         const T &value,
                                         command_queue &queue)
{
    return ::boost::compute::copy_async(
               ::boost::compute::make_constant_iterator(value, 0),
               ::boost::compute::make_constant_iterator(value, count),
               first,
               queue
           );
}

#if defined(CL_VERSION_1_2)

// meta-function returing true if Iterator points to a range of values
// that can be filled using clEnqueueFillBuffer(). to meet this criteria
// it must have a buffer accessible through iter.get_buffer() and the
// size of its value_type must by in {1, 2, 4, 8, 16, 32, 64, 128}.
template<class Iterator>
struct is_valid_fill_buffer_iterator :
    public mpl::and_<
        is_buffer_iterator<Iterator>,
        mpl::contains<
            mpl::vector<
                mpl::int_<1>,
                mpl::int_<2>,
                mpl::int_<4>,
                mpl::int_<8>,
                mpl::int_<16>,
                mpl::int_<32>,
                mpl::int_<64>,
                mpl::int_<128>
            >,
            mpl::int_<
                sizeof(typename std::iterator_traits<Iterator>::value_type)
            >
        >
    >::type { };

// specialization which uses clEnqueueFillBuffer for buffer iterators
template<class BufferIterator, class T>
inline void
dispatch_fill(BufferIterator first,
              size_t count,
              const T &value,
              command_queue &queue,
              typename boost::enable_if<
                 is_valid_fill_buffer_iterator<BufferIterator>
              >::type* = 0)
{
    typedef typename std::iterator_traits<BufferIterator>::value_type value_type;

    value_type pattern = static_cast<value_type>(value);
    size_t offset = static_cast<size_t>(first.get_index());

    queue.enqueue_fill_buffer(first.get_buffer(),
                              &pattern,
                              sizeof(value_type),
                              offset * sizeof(value_type),
                              count * sizeof(value_type));
}

template<class BufferIterator, class T>
inline future<void>
dispatch_fill_async(BufferIterator first,
                    size_t count,
                    const T &value,
                    command_queue &queue,
                    typename boost::enable_if<
                       is_valid_fill_buffer_iterator<BufferIterator>
                    >::type* = 0)
{
    typedef typename std::iterator_traits<BufferIterator>::value_type value_type;

    value_type pattern = static_cast<value_type>(value);
    size_t offset = static_cast<size_t>(first.get_index());

    event event_ =
        queue.enqueue_fill_buffer(first.get_buffer(),
                                  &pattern,
                                  sizeof(value_type),
                                  offset * sizeof(value_type),
                                  count * sizeof(value_type));

    return future<void>(event_);
}

// default implementations
template<class BufferIterator, class T>
inline void
dispatch_fill(BufferIterator first,
              size_t count,
              const T &value,
              command_queue &queue,
              typename boost::disable_if<
                  is_valid_fill_buffer_iterator<BufferIterator>
              >::type* = 0)
{
    fill_with_copy(first, count, value, queue);
}

template<class BufferIterator, class T>
inline future<void>
dispatch_fill_async(BufferIterator first,
                    size_t count,
                    const T &value,
                    command_queue &queue,
                    typename boost::disable_if<
                        is_valid_fill_buffer_iterator<BufferIterator>
                    >::type* = 0)
{
    return fill_async_with_copy(first, count, value, queue);
}
#else
template<class BufferIterator, class T>
inline void dispatch_fill(BufferIterator first,
                          size_t count,
                          const T &value,
                          command_queue &queue)
{
    fill_with_copy(first, count, value, queue);
}

template<class BufferIterator, class T>
inline future<void> dispatch_fill_async(BufferIterator first,
                                        size_t count,
                                        const T &value,
                                        command_queue &queue)
{
    return fill_async_with_copy(first, count, value, queue);
}
#endif // !defined(CL_VERSION_1_2)

} // end detail namespace

/// Fills the range [\p first, \p last) with \p value.
///
/// \see fill_n()
template<class BufferIterator, class T>
inline void fill(BufferIterator first,
                 BufferIterator last,
                 const T &value,
                 command_queue &queue = system::default_queue())
{
    size_t count = detail::iterator_range_size(first, last);

    detail::dispatch_fill(first, count, value, queue);
}

template<class BufferIterator, class T>
inline future<void> fill_async(BufferIterator first,
                               BufferIterator last,
                               const T &value,
                               command_queue &queue = system::default_queue())
{
    size_t count = detail::iterator_range_size(first, last);

    return detail::dispatch_fill_async(first, count, value, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_FILL_HPP
