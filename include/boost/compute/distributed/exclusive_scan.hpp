//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_EXCLUSIVE_SCAN_HPP
#define BOOST_COMPUTE_DISTRIBUTED_EXCLUSIVE_SCAN_HPP

#include <vector>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/algorithm/merge.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>

#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/vector.hpp>

namespace boost {
namespace compute {
namespace distributed {

template<
    class InputType, weight_func weight, class Alloc,
    class OutputType,
    class BinaryOperator
>
inline void
exclusive_scan(const vector<InputType, weight, Alloc> &input,
               vector<OutputType, weight, Alloc> &result,
               OutputType init,
               BinaryOperator binary_op,
               command_queue &queue)
{
    BOOST_ASSERT(input.parts() == result.parts());
    BOOST_ASSERT(input.size() == result.size());

    std::vector<OutputType> input_tails;
    input_tails.reserve(input.parts() - 1);
    for(size_t i = 0; i < input.parts(); i++)
    {
        if(input.begin(i) != input.end(i) && i < (input.parts() - 1))
        {
            input_tails.push_back(
                static_cast<OutputType>(
                    (input.end(i) - 1).read(queue.get(i))
                )
            );
        }

        if(i == 0)
        {
            ::boost::compute::exclusive_scan(
                input.begin(i),
                input.end(i),
                result.begin(i),
                init,
                binary_op,
                queue.get(i)
            );
        }
        else
        {
            ::boost::compute::exclusive_scan(
                input.begin(i),
                input.end(i),
                result.begin(i),
                input_tails[i - 1],
                binary_op,
                queue.get(i)
            );
        }
    }

    // find device for calculating partial sum of last elements of input vector
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

    std::vector<OutputType> output_tails(input_tails.size());
    for(size_t i = 0; i < input.parts() - 1; i++)
    {
        if(input.begin(i) != input.end(i))
        {
            output_tails[i] = (result.end(i) - 1).read(queue.get(i));
        }
    }
    ::boost::compute::vector<OutputType> output_tails_device(
        output_tails.size(), device_queue.get_context()
    );
    ::boost::compute::copy_async(
        output_tails.begin(),
        output_tails.end(),
        output_tails_device.begin(),
        device_queue
    );
    ::boost::compute::inclusive_scan(
        output_tails_device.begin(),
        output_tails_device.end(),
        output_tails_device.begin(),
        device_queue
    );
    ::boost::compute::copy(
        output_tails_device.begin(),
        output_tails_device.end(),
        output_tails.begin(),
        device_queue
    );
    for(size_t i = 1; i < input.parts(); i++)
    {
        ::boost::compute::transform(
            result.begin(i),
            result.end(i),
            ::boost::compute::make_constant_iterator(
                output_tails[i - 1]
            ),
            result.begin(i),
            binary_op,
            queue.get(i)
        );
    }
}

/// \overload
template<
    class InputType, weight_func weight, class Alloc,
    class OutputType
>
inline void
exclusive_scan(const vector<InputType, weight, Alloc> &input,
               vector<OutputType, weight, Alloc> &result,
               OutputType init,
               command_queue &queue)
{
    ::boost::compute::distributed::exclusive_scan(
        input,
        result,
        init,
        boost::compute::plus<OutputType>(),
        queue
    );
}

/// \overload
template<
    class InputType, weight_func weight, class Alloc,
    class OutputType
>
inline void
exclusive_scan(const vector<InputType, weight, Alloc> &input,
               vector<OutputType, weight, Alloc> &result,
               command_queue &queue)
{
    ::boost::compute::distributed::exclusive_scan(
        input,
        result,
        OutputType(0),
        boost::compute::plus<OutputType>(),
        queue
    );
}

} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_SCAN_HPP */
