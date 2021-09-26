//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_TEST_CHECK_FUNCTIONS_HPP
#define BOOST_COMPUTE_TEST_CHECK_FUNCTIONS_HPP

#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/vector.hpp>
#include <boost/compute/distributed/copy.hpp>

template<
    class T1, boost::compute::distributed::weight_func weight, class Alloc1,
    class T2, class Alloc2
>
inline bool
distributed_equal(const boost::compute::distributed::vector<T1, weight, Alloc1> &input1,
                  const boost::compute::distributed::vector<T2, weight, Alloc2> &input2,
                  boost::compute::distributed::command_queue &queue)
{
    if(input1.parts() != input2.parts()) {
        return false;
    }
    if(input1.size() != input2.size()) {
        return false;
    }

    for(size_t i = 0; i < input1.parts(); i++)
    {
        if(
            !boost::compute::equal(
                input1.begin(i), input1.end(i), input2.begin(i), queue.get(i)
            )
        )
        {
            return false;
        }
    }
    return true;
}

template<
    class T, boost::compute::distributed::weight_func weight, class Alloc
>
inline bool
distributed_equal(const boost::compute::distributed::vector<T, weight, Alloc> &input,
                  const T value,
                  boost::compute::distributed::command_queue &queue)
{
    for(size_t i = 0; i < input.parts(); i++)
    {
        if(
            !boost::compute::equal(
                input.begin(i),
                input.end(i),
                boost::compute::make_constant_iterator(value),
                queue.get(i)
            )
        )
        {
            return false;
        }
    }
    return true;
}

template<
    class T, boost::compute::distributed::weight_func weight, class Alloc
>
inline bool
distributed_equal(const boost::compute::distributed::vector<T, weight, Alloc> &input,
                  typename std::vector<T>::iterator first,
                  typename std::vector<T>::iterator end,
                  boost::compute::distributed::command_queue &queue)
{
    if(std::distance(first, end) != input.size()) {
        return false;
    }

    typename std::vector<T>::iterator part_first = first;
    typename std::vector<T>::iterator part_end = first;
    for(size_t i = 0; i < input.parts(); i++)
    {
        part_end += input.part_size(i);
        boost::compute::vector<T> data(part_first, part_end, queue.get(i));
        if(
            !boost::compute::equal(
                input.begin(i),
                input.end(i),
                data.begin(),
                queue.get(i)
            )
        )
        {
            return false;
        }
        part_first = part_end;
    }
    return true;
}

template<
    class T1, boost::compute::distributed::weight_func weight1, class Alloc1,
    class T2, boost::compute::distributed::weight_func weight2, class Alloc2
>
inline bool
distributed_equal(const boost::compute::distributed::vector<T1, weight1, Alloc1> &input1,
                  const boost::compute::distributed::vector<T2, weight2, Alloc2> &input2,
                  boost::compute::distributed::command_queue &queue1,
                  boost::compute::distributed::command_queue &queue2)
{
    if(input1.parts() != input2.parts()) {
        return false;
    }
    if(input1.size() != input2.size()) {
        return false;
    }

    std::vector<T2> host1(input1.size());
    boost::compute::distributed::copy(input1, host1.begin(), queue1);
    return distributed_equal(input2, host1.begin(), host1.end(), queue2);
}

#endif /* BOOST_COMPUTE_TEST_TEST_CHECK_FUNCTIONS_HPP */
