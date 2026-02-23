//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_COPY_HPP
#define BOOST_COMPUTE_DISTRIBUTED_COPY_HPP

#include <algorithm>
#include <iterator>

#include <boost/utility/enable_if.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/or.hpp>

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/wait_list.hpp>

#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/detail/weight_func.hpp>
#include <boost/compute/distributed/detail/is_distributed_vector.hpp>

namespace boost {
namespace compute {
namespace distributed {

// forward declaration for distributed::vector<T, weight_func, Alloc>
template<
    class T,
    weight_func weight,
    class Alloc
>
class vector;

// host -> distributed::vector
/// Copies the values in the range [\p first, \p last) allocated on host to
/// distributed::vector \p result. The copy is performed asynchronously.
template <
    class InputIterator,
    class T, weight_func weight, class Alloc
>
inline std::vector<event>
copy_async(InputIterator first,
           InputIterator last,
           vector<T, weight, Alloc> &result,
           command_queue &queue,
           typename boost::enable_if_c<
               !is_device_iterator<InputIterator>::value
           >::type* = 0)
{
    typedef typename
        std::iterator_traits<InputIterator>::difference_type diff_type;

    std::vector<event> events;
    events.reserve(result.parts());

    InputIterator part_first = first;
    InputIterator part_end = first;
    for(size_t i = 0; i < result.parts(); i++)
    {
        part_end = (std::min)(
            part_end + static_cast<diff_type>(result.part_size(i)),
            last
        );
        event e =
            ::boost::compute::copy_async(
                part_first,
                part_end,
                result.begin(i),
                queue.get(i)
            ).get_event();
        if(e.get()) {
            events.push_back(e);
        }
        part_first = part_end;
    }
    return events;
}

// host -> distributed::vector
/// Copies the values in the range [\p first, \p last) allocated on host to
/// distributed::vector \p result.
template <
    class InputIterator,
    class T, weight_func weight, class Alloc
>
inline void
copy(InputIterator first,
     InputIterator last,
     vector<T, weight, Alloc> &result,
     command_queue &queue,
     typename boost::enable_if_c<
         !is_device_iterator<InputIterator>::value
     >::type* = 0)
{
    std::vector<event> events =
        ::boost::compute::distributed::copy_async(first, last, result, queue);
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
}

// distributed::vector -> host
/// Copy all values from distributed::vector \p input to the range beginning at
/// \p result allocated on the host.
template <
    class T, weight_func weight, class Alloc,
    class OutputIterator
>
inline std::vector<event>
copy_async(const vector<T, weight, Alloc> &input,
           OutputIterator result,
           command_queue &queue,
           typename boost::enable_if<
               mpl::and_<
                   mpl::not_<
                       is_device_iterator<OutputIterator>
                   >,
                   mpl::not_<
                       detail::is_distributed_vector<OutputIterator>
                   >
               >
           >::type* = 0)
{
    typedef typename
        std::iterator_traits<OutputIterator>::difference_type diff_type;

    std::vector<event> events;
    events.reserve(input.parts());

    OutputIterator part_result = result;
    for(size_t i = 0; i < input.parts(); i++)
    {
        event e =
            ::boost::compute::copy_async(
                input.begin(i),
                input.end(i),
                part_result,
                queue.get(i)
            ).get_event();
        if(e.get()) {
            events.push_back(e);
        }
        part_result += static_cast<diff_type>(input.part_size(i));
    }
    return events;
}

// distributed::vector -> host
/// Copy all values from distributed::vector \p input to the range beginning at
/// \p result allocated on the host.
template <
    class T, weight_func weight, class Alloc,
    class OutputIterator
>
inline void
copy(const vector<T, weight, Alloc> &input,
     OutputIterator result,
     command_queue &queue,
     typename boost::enable_if<
         mpl::and_<
             mpl::not_<
                 is_device_iterator<OutputIterator>
             >,
             mpl::not_<
                 detail::is_distributed_vector<OutputIterator>
             >
         >
     >::type* = 0)
{
    std::vector<event> events =
        ::boost::compute::distributed::copy_async(input, result, queue);
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
}

// device -> distributed::vector
/// Copies the values in the range [\p first, \p last) allocated on an OpenCL
/// device to the distributed::vector \p result. The copy is performed
/// asynchronously.
template <
    class InputIterator,
    class T, weight_func weight, class Alloc
>
inline std::vector<event>
copy_async(InputIterator first,
           InputIterator last,
           vector<T, weight, Alloc> &result,
           ::boost::compute::command_queue &device_queue,
           command_queue &distributed_queue,
           typename boost::enable_if_c<
               is_device_iterator<InputIterator>::value
           >::type* = 0)
{
    typedef typename
        std::iterator_traits<InputIterator>::difference_type diff_type;

    std::vector<event> events;
    events.reserve(result.parts());

    InputIterator part_first = first;
    InputIterator part_end = first;
    for(size_t i = 0; i < result.parts(); i++)
    {
        BOOST_ASSERT_MSG(
            distributed_queue.get(i).get_context() == device_queue.get_context(),
            "copy_async() is only supported when context of every queue in"
            " distributed_queue is the same context as context of device_queue"
        );
        part_end = (std::min)(
            part_end + static_cast<diff_type>(result.part_size(i)),
            last
        );
        event e =
            ::boost::compute::copy_async(
                part_first,
                part_end,
                result.begin(i),
                distributed_queue.get(i)
            ).get_event();
        if(e.get()) {
            events.push_back(e);
        }
        part_first = part_end;
    }
    return events;
}

// host -> distributed::vector
/// Copies the values in the range [\p first, \p last) allocated on an OpenCL
/// device to the distributed::vector \p result.
template <
    class InputIterator,
    class T, weight_func weight, class Alloc
>
inline void
copy(InputIterator first,
     InputIterator last,
     vector<T, weight, Alloc> &result,
     ::boost::compute::command_queue &device_queue,
     command_queue &distributed_queue,
     typename boost::enable_if_c<
         is_device_iterator<InputIterator>::value
     >::type* = 0)
{
    typedef typename
        std::iterator_traits<InputIterator>::difference_type diff_type;

    std::vector<event> events;
    events.reserve(result.parts());

    InputIterator part_first = first;
    InputIterator part_end = first;
    for(size_t i = 0; i < result.parts(); i++)
    {
        part_end = (std::min)(
            part_end + static_cast<diff_type>(result.part_size(i)),
            last
        );
        if(distributed_queue.get(i).get_context() == device_queue.get_context())
        {
            event e =
                ::boost::compute::copy_async(
                    part_first,
                    part_end,
                    result.begin(i),
                    distributed_queue.get(i)
                ).get_event();
            if(e.get()) {
                events.push_back(e);
            }
        }
        else {
            std::vector<T> host(result.part_size(i));
            ::boost::compute::copy(
                part_first,
                part_end,
                host.begin(),
                device_queue
            );
            ::boost::compute::copy(
                host.begin(),
                host.end(),
                result.begin(i),
                distributed_queue.get(i)
            );
        }
        part_first = part_end;
    }
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
}

// distributed::vector -> device
/// Copy all values from distributed::vector \p input to the range beginning at
/// \p result allocated on an OpenCL device.
template <
    class T, weight_func weight, class Alloc,
    class OutputIterator
>
inline std::vector<event>
copy_async(const vector<T, weight, Alloc> &input,
           OutputIterator result,
           command_queue &distributed_queue,
           ::boost::compute::command_queue &device_queue,
           typename boost::enable_if<
               mpl::and_<
                   is_device_iterator<OutputIterator>,
                   mpl::not_<
                       detail::is_distributed_vector<OutputIterator>
                   >
               >
           >::type* = 0)
{
    typedef typename
        std::iterator_traits<OutputIterator>::difference_type diff_type;

    std::vector<event> events;
    events.reserve(input.parts());

    OutputIterator part_result = result;
    for(size_t i = 0; i < input.parts(); i++)
    {
        BOOST_ASSERT_MSG(
            distributed_queue.get(i).get_context() == device_queue.get_context(),
            "copy_async() is only supported when context of every queue in"
            " distributed_queue is the same context as context of device_queue"
        );
        event e =
            ::boost::compute::copy_async(
                input.begin(i),
                input.end(i),
                part_result,
                distributed_queue.get(i)
            ).get_event();
        if(e.get()) {
            events.push_back(e);
        }
        part_result += static_cast<diff_type>(input.part_size(i));
    }
    return events;
}

// distributed::vector -> device
/// Copy all values from distributed::vector \p input to the range beginning at
/// \p result allocated on an OpenCL device.
template <
    class T, weight_func weight, class Alloc,
    class OutputIterator
>
inline void
copy(const vector<T, weight, Alloc> &input,
     OutputIterator result,
     command_queue &distributed_queue,
     ::boost::compute::command_queue &device_queue,
     typename boost::enable_if<
         mpl::and_<
             is_device_iterator<OutputIterator>,
             mpl::not_<
                 detail::is_distributed_vector<OutputIterator>
             >
         >
     >::type* = 0)
{
    typedef typename
        std::iterator_traits<OutputIterator>::difference_type diff_type;

    std::vector<event> events;
    events.reserve(input.parts());

    OutputIterator part_result = result;
    for(size_t i = 0; i < input.parts(); i++)
    {
        if(distributed_queue.get(i).get_context() == device_queue.get_context())
        {
            event e =
                ::boost::compute::copy_async(
                    input.begin(i),
                    input.end(i),
                    part_result,
                    distributed_queue.get(i)
                ).get_event();
            if(e.get()) {
                events.push_back(e);
            }
        }
        else {
            std::vector<T> host(input.part_size(i));
            ::boost::compute::copy(
                input.begin(i),
                input.end(i),
                host.begin(),
                distributed_queue.get(i)
            );
            ::boost::compute::copy(
                host.begin(),
                host.end(),
                part_result,
                device_queue
            );
        }
        part_result += static_cast<diff_type>(input.part_size(i));
    }
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
}

// distributed::vector -> distributed::vector
/// Copy distributed::vector \p input into \p output. The copy is performed
/// asynchronously.
///
/// Both vectors must be able to use \p queue, have the same weight function,
/// the same size and the same number of parts.
template<
    class T1, weight_func weight, class Alloc1,
    class T2, class Alloc2
>
inline std::vector<event>
copy_async(const vector<T1, weight, Alloc1> &input,
           vector<T2, weight, Alloc2> &output,
           command_queue &queue)
{
    BOOST_ASSERT(input.parts() == output.parts());
    BOOST_ASSERT(input.size() == output.size());

    std::vector<event> events;
    events.reserve(input.parts());

    for(size_t i = 0; i < input.parts(); i++)
    {
        event e =
            ::boost::compute::copy_async(
                input.begin(i),
                input.end(i),
                output.begin(i),
                queue.get(i)
            ).get_event();
        if(e.get()) {
            events.push_back(e);
        }
    }
    return events;
}

// distributed::vector -> distributed::vector
/// Copy distributed::vector \p input into \p output.
template<
    class T1, weight_func weight1, class Alloc1,
    class T2, weight_func weight2, class Alloc2
>
inline void
copy(const vector<T1, weight1, Alloc1> &input,
     vector<T2, weight2, Alloc2> &output,
     command_queue &input_queue,
     command_queue &output_queue)
{
    std::vector<event> events;
    events.reserve(input.parts());

    std::vector<T2> host(input.size());
    events =
        ::boost::compute::distributed::copy_async(
            input, host.begin(), input_queue
         );
    // wait for copying from input to host
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
    // copy from host to output vector
    events =
        ::boost::compute::distributed::copy_async(
            host.begin(), host.end(), output, output_queue
    );
    for(size_t i = 0; i < events.size(); i++) {
        events[i].wait();
    }
}

// distributed::vector -> distributed::vector
/// Copy distributed::vector \p input into \p output.
///
/// Both vectors must be able to use \p queue, have the same weight function,
/// and have the same number of parts.
template<
    class T1, weight_func weight, class Alloc1,
    class T2, class Alloc2
>
inline void
copy(const vector<T1, weight, Alloc1> &input,
     vector<T2, weight, Alloc2> &output,
     command_queue &queue)
{
    BOOST_ASSERT(input.parts() == output.parts());
    if(input.size() == output.size()) {
        std::vector<event> events =
            ::boost::compute::distributed::copy_async(
                input, output, queue
            );
        for(size_t i = 0; i < events.size(); i++) {
              events[i].wait();
        }
    }
    else {
        ::boost::compute::distributed::copy(
            input, output, queue, queue
        );
    }
}

} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_COPY_HPP */
