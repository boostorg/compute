//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_TRANSFORM_HPP
#define BOOST_COMPUTE_DISTRIBUTED_TRANSFORM_HPP

#include <boost/compute/iterator/transform_iterator.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/functional/detail/unpack.hpp>

#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/vector.hpp>

namespace boost {
namespace compute {
namespace distributed {

/// Transforms all the elements from vector \p input using operator \p op
/// and stores the results in \p result. The transform is performed
/// asynchronously and it returns a vector of events, each assisted with
/// transformation of successive parts of \p input.
///
/// Distributed command queue \p queue has to span same set of compute devices
/// (including their contexts) as distributed command queue used to create
/// \p input and \p output vectors.
///
/// \see distributed::transform()
template<
    class InputType, class OutputType,
    weight_func weight, class Alloc,
    class UnaryOperator
>
inline std::vector<event>
transform_async(const vector<InputType, weight, Alloc> &input,
                vector<OutputType, weight, Alloc> &result,
                UnaryOperator op,
                command_queue &queue)
{
    BOOST_ASSERT(input.parts() == result.parts());
    BOOST_ASSERT(input.size() == result.size());

    std::vector<event> events;
    events.reserve(input.parts());

    for(size_t i = 0; i < input.parts(); i++)
    {
        event e =
            ::boost::compute::copy_async(
                ::boost::compute::make_transform_iterator(input.begin(i), op),
                ::boost::compute::make_transform_iterator(input.end(i), op),
                result.begin(i),
                queue.get(i)
            ).get_event();
        if(e.get()) {
            events.push_back(e);
        }
    }
    return events;
}

/// \overload
template<
    class InputType1, class InputType2, class OutputType,
    weight_func weight, class Alloc,
    class BinaryOperator
>
inline std::vector<event>
transform_async(const vector<InputType1, weight, Alloc> &input1,
                const vector<InputType2, weight, Alloc> &input2,
                vector<OutputType, weight, Alloc> &result,
                BinaryOperator op,
                command_queue &queue)
{
    BOOST_ASSERT(input1.parts() == input2.parts());
    BOOST_ASSERT(input1.parts() == result.parts());
    BOOST_ASSERT(input1.size() == input1.size());
    BOOST_ASSERT(input1.size() == result.size());

    std::vector<event> events;
    events.reserve(input1.parts());

    ::boost::compute::detail::unpacked<BinaryOperator> unpacked_op =
        ::boost::compute::detail::unpack(op);
    for(size_t i = 0; i < input1.parts(); i++)
    {
        event e =
            ::boost::compute::copy_async(
                ::boost::compute::make_transform_iterator(
                    ::boost::compute::make_zip_iterator(
                        boost::make_tuple(input1.begin(i), input2.begin(i))
                    ),
                    unpacked_op
                ),
                ::boost::compute::make_transform_iterator(
                    ::boost::compute::make_zip_iterator(
                        boost::make_tuple(input1.end(i), input2.end(i))
                    ),
                    unpacked_op
                ),
                result.begin(i),
                queue.get(i)
            ).get_event();
        if(e.get()) {
            events.push_back(e);
        }
    }
    return events;
}

/// Transforms all the elements from vector \p input using operator \p op
/// and stores the results in \p result.
///
/// Distributed command queue \p queue has to span same set of compute devices
/// (including their contexts) as distributed command queue used to create
/// \p input and \p output vectors.
///
/// \see  distributed::transform_async()
template<
    class InputType, weight_func weight, class Alloc, class OutputType,
    class UnaryOperator
>
inline void
transform(const vector<InputType, weight, Alloc> &input,
          vector<OutputType, weight, Alloc> &output,
          UnaryOperator op,
          command_queue &queue)
{
    std::vector<event> events =
        transform_async(
            input, output, op, queue
        );
    for(size_t i = 0; i < events.size(); i++) {
          events[i].wait();
    }
}

/// \overload
template<
    class InputType1, class InputType2, class OutputType,
    weight_func weight, class Alloc,
    class BinaryOperator
>
inline void
transform(const vector<InputType1, weight, Alloc> &input1,
          const vector<InputType2, weight, Alloc> &input2,
          vector<OutputType, weight, Alloc> &output,
          BinaryOperator op,
          command_queue &queue)
{
    std::vector<event> events =
        transform_async(
            input1, input2, output, op, queue
        );
    for(size_t i = 0; i < events.size(); i++) {
          events[i].wait();
    }
}

} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_TRANSFORM_HPP */
