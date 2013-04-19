//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestTransformReduce
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/transform_reduce.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(sum_abs_int)
{
    int data[] = { 1, -2, -3, -4, 5 };
    bc::vector<int> vector(data, data + 5, context);

    int sum =
        bc::transform_reduce(vector.begin(),
                             vector.end(),
                             bc::abs<int>(),
                             0,
                             bc::plus<int>(),
                             queue);
    BOOST_CHECK_EQUAL(sum, 15);
}

BOOST_AUTO_TEST_CASE(multiply_vector_length)
{
    float data[] = { 2.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 3.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 4.0f, 0.0f };
    bc::vector<bc::float4_> vector(reinterpret_cast<bc::float4_ *>(data),
                                   reinterpret_cast<bc::float4_ *>(data) + 3,
                                   context);

    float product =
        bc::transform_reduce(vector.begin(),
                             vector.end(),
                             bc::length<bc::float4_>(),
                             1.0f,
                             bc::multiplies<float>(),
                             queue);
    BOOST_CHECK_CLOSE(product, 24.0f, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()
