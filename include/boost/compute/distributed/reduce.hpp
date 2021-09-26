//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_REDUCE_HPP
#define BOOST_COMPUTE_DISTRIBUTED_REDUCE_HPP

#include <vector>

#include <boost/utility/enable_if.hpp>

#include <boost/compute/buffer.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/algorithm/copy_n.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/type_traits/is_device_iterator.hpp>

#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/vector.hpp>

namespace boost {
namespace compute {
namespace distributed {

namespace detail {

template<class OutputIterator>
inline ::boost::compute::command_queue&
final_reduce_queue(OutputIterator result,
                   command_queue &queue,
                   typename boost::enable_if_c<
                       !is_device_iterator<OutputIterator>::value
                   >::type* = 0)
{
    (void) result;

    ::boost::compute::command_queue& device_queue = queue.get(0);
    // CPU device is preferred, however if there is none, the first device
    // queue is used
    for(size_t i = 0; i < queue.size(); i++)
    {
        if(queue.get(i).get_device().type() & ::boost::compute::device::cpu)
        {
            device_queue = queue.get(i);
            break;
        }
    }
    return device_queue;
}

template<class OutputIterator>
inline ::boost::compute::command_queue&
final_reduce_queue(OutputIterator result,
                   command_queue &queue,
                   typename boost::enable_if_c<
                       is_device_iterator<OutputIterator>::value
                   >::type* = 0)
{
    // first, find all queues that can be used with result iterator
    const ::boost::compute::context& result_context =
        result.get_buffer().get_context();
    std::vector<size_t> compatible_queues;
    for(size_t i = 0; i < queue.size(); i++)
    {
        if(queue.get(i).get_context() == result_context)
        {
            compatible_queues.push_back(i);
        }
    }
    BOOST_ASSERT_MSG(
        !compatible_queues.empty(),
        "There is no device command queue that can be use to copy to result"
    );

    // then choose device queue from compatible device queues

    // CPU device is preferred, however if there is none, the first
    // compatible device queue is used
    ::boost::compute::command_queue& device_queue = queue.get(compatible_queues[0]);
    for(size_t i = 0; i < compatible_queues.size(); i++)
    {
        size_t n = compatible_queues[i];
        if(queue.get(n).get_device().type() & ::boost::compute::device::cpu)
        {
            device_queue = queue.get(n);
            break;
        }
    }
    return device_queue;
}

template<
    class InputType, weight_func weight, class Alloc,
    class OutputIterator,
    class BinaryFunction
>
inline void
dispatch_reduce(const vector<InputType, weight, Alloc> &input,
                OutputIterator result,
                BinaryFunction function,
                command_queue &queue)
{
    typedef typename
        boost::compute::result_of<BinaryFunction(InputType, InputType)>::type
        result_type;

    // find device queue for the final reduction
    ::boost::compute::command_queue& device_queue =
        final_reduce_queue(result, queue);

    ::boost::compute::buffer parts_results_device(
        device_queue.get_context(), input.parts() * sizeof(result_type)
    );

    // if all devices queues are in the same OpenCL context we can
    // save part reduction directly into parts_results_device buffer
    size_t reduced = 0;
    if(queue.one_context())
    {
        // reduce each part of input vector
        for(size_t i = 0; i < input.parts(); i++)
        {
            if(input.begin(i) != input.end(i))
            {
                // async, because it stores result on device
                ::boost::compute::reduce(
                    input.begin(i),
                    input.end(i),
                    ::boost::compute::make_buffer_iterator<result_type>(
                        parts_results_device, reduced
                    ),
                    function,
                    queue.get(i)
                );
                reduced++;
            }
        }
        // add marker on every queue that is not device_queue, because
        // we need to know when reductions are done
        wait_list reduce_markers;
        reduce_markers.reserve(reduced);
        for(size_t i = 0; i < input.parts(); i++)
        {
            if(input.begin(i) != input.end(i) && queue.get(i) != device_queue)
            {
                reduce_markers.insert(queue.get(i).enqueue_marker());
            }
        }
        // if it is possible we enqueue a barrier in device_queue which waits
        // for reduce_markers (we can do this since all events are in the same
        // context); otherwise, we need to sync. wait for those events
        #ifdef CL_VERSION_1_2
        if(device_queue.check_device_version(1, 2)) {
            device_queue.enqueue_barrier(reduce_markers);
        }
        #endif
        {
            reduce_markers.wait();
        }
    }
    else
    {
        // reduce each part of input vector
        std::vector<result_type> parts_results_host(input.parts());
        for(size_t i = 0; i < input.parts(); i++)
        {
            if(input.begin(i) != input.end(i))
            {
                // sync, because it stores result on host
                ::boost::compute::reduce(
                    input.begin(i),
                    input.end(i),
                    &parts_results_host[reduced],
                    function,
                    queue.get(i)
                );
                reduced++;
            }
        }
        // sync, because it copies from host to device
        ::boost::compute::copy_n(
            parts_results_host.begin(),
            reduced,
            ::boost::compute::make_buffer_iterator<result_type>(
                parts_results_device
            ),
            device_queue
        );
    }
    // final reduction
    // async if result is device_iterator, sync otherwise
    ::boost::compute::reduce(
        ::boost::compute::make_buffer_iterator<result_type>(
            parts_results_device
        ),
        ::boost::compute::make_buffer_iterator<result_type>(
            parts_results_device, reduced
        ),
        result,
        function,
        device_queue
    );
}

// special case for when OutputIterator is a host iterator
// and binary operator is plus<T>
template<
    class InputType, weight_func weight, class Alloc,
    class OutputIterator,
    class T
>
inline void
dispatch_reduce(const vector<InputType, weight, Alloc> &input,
                OutputIterator result,
                ::boost::compute::plus<T> function,
                command_queue &queue,
                typename boost::enable_if_c<
                    !is_device_iterator<OutputIterator>::value
                >::type* = 0)
{
    // reduce each part of input vector
    std::vector<T> parts_results_host(input.parts());
    for(size_t i = 0; i < input.parts(); i++)
    {
        ::boost::compute::reduce(
            input.begin(i),
            input.end(i),
            &parts_results_host[i],
            function,
            queue.get(i)
        );
    }

    // final reduction
    *result = parts_results_host[0];
    for(size_t i = 1; i < input.parts(); i++)
    {
        *result += static_cast<T>(parts_results_host[i]);
    }
}

// special case for when OutputIterator is a host iterator
// and binary operator is min<T>
template<
    class InputType, weight_func weight, class Alloc,
    class OutputIterator,
    class T
>
inline void
dispatch_reduce(vector<InputType, weight, Alloc> &input,
                OutputIterator result,
                ::boost::compute::min<T> function,
                command_queue &queue,
                typename boost::enable_if_c<
                    !is_device_iterator<OutputIterator>::value
                >::type* = 0)
{
    // reduce each part of input vector
    std::vector<T> parts_results_host(input.parts());
    for(size_t i = 0; i < input.parts(); i++)
    {
        ::boost::compute::reduce(
            input.begin(i),
            input.end(i),
            &parts_results_host[i],
            function,
            queue.get(i)
        );
    }

    // final reduction
    *result = parts_results_host[0];
    for(size_t i = 1; i < input.parts(); i++)
    {
        *result = (std::min)(static_cast<T>(*result), parts_results_host[i]);
    }
}

// special case for when OutputIterator is a host iterator
// and binary operator is max<T>
template<
    class InputType, weight_func weight, class Alloc,
    class OutputIterator,
    class T
>
inline void
dispatch_reduce(const vector<InputType, weight, Alloc> &input,
                OutputIterator result,
                ::boost::compute::max<T> function,
                command_queue &queue,
                typename boost::enable_if_c<
                    !is_device_iterator<OutputIterator>::value
                >::type* = 0)
{
    // reduce each part of input vector
    std::vector<T> parts_results_host(input.parts());
    for(size_t i = 0; i < input.parts(); i++)
    {
        ::boost::compute::reduce(
            input.begin(i),
            input.end(i),
            &parts_results_host[i],
            function,
            queue.get(i)
        );
    }

    // final reduction
    *result = parts_results_host[0];
    for(size_t i = 1; i < input.parts(); i++)
    {
        *result = (std::max)(static_cast<T>(*result), parts_results_host[i]);
    }
}

} // end detail namespace

/// Returns the result of applying \p function to the elements in the
/// \p input vector.
///
/// If no function is specified, \c plus will be used.
///
/// \param input input vector
/// \param result iterator pointing to the output
/// \param function binary reduction function
/// \param queue distributed command queue to perform the operation
///
/// Distributed command queue \p queue has to span same set of compute devices
/// (including their contexts) as distributed command queue used to create
///  \p input vector.
///
/// If \p result is a device iterator, its underlying buffer must be allocated
/// in context of at least one device command queue from \p queue.
///
/// The \c reduce() algorithm assumes that the binary reduction function is
/// associative. When used with non-associative functions the result may
/// be non-deterministic and vary in precision. Notably this affects the
/// \c plus<float>() function as floating-point addition is not associative
/// and may produce slightly different results than a serial algorithm.
///
/// This algorithm supports both host and device iterators for the
/// result argument. This allows for values to be reduced and copied
/// to the host all with a single function call.
template<
    class InputType, weight_func weight, class Alloc,
    class OutputIterator,
    class BinaryFunction
>
inline void
reduce(const vector<InputType, weight, Alloc> &input,
       OutputIterator result,
       BinaryFunction function,
       command_queue &queue)
{
    if(input.empty()) {
        return;
    }

    if(input.parts() == 1) {
        ::boost::compute::reduce(
            input.begin(0),
            input.end(0),
            result,
            function,
            queue.get(0)
        );
        return;
    }
    detail::dispatch_reduce(input, result, function, queue);
}

/// \overload
template<
    class InputType, weight_func weight, class Alloc,
    class OutputIterator
>
inline void
reduce(const vector<InputType, weight, Alloc> &input,
       OutputIterator result,
       command_queue &queue)
{
    return reduce(input, result, ::boost::compute::plus<InputType>(), queue);
}

} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_REDUCE_HPP */
