//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestClosure
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/closure.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(add_two)
{
    int two = 2;
    BOOST_COMPUTE_CLOSURE(int, add_two, (int x), (two),
    {
        return x + two;
    });

    int data[] = { 1, 2, 3, 4 };
    compute::vector<int> vector(data, data + 4, queue);

    compute::transform(
        vector.begin(), vector.end(), vector.begin(), add_two, queue
    );
    CHECK_RANGE_EQUAL(int, 4, vector, (3, 4, 5, 6));
}

BOOST_AUTO_TEST_CASE(add_two_and_pi)
{
    int two = 2;
    float pi = 3.14f;
    BOOST_COMPUTE_CLOSURE(float, add_two_and_pi, (float x), (two, pi),
    {
        return x + two + pi;
    });

    float data[] = { 1.9f, 2.2f, 3.4f, 4.7f };
    compute::vector<float> vector(data, data + 4, queue);

    compute::transform(
        vector.begin(), vector.end(), vector.begin(), add_two_and_pi, queue
    );

    std::vector<float> results(4);
    compute::copy(vector.begin(), vector.end(), results.begin(), queue);
    BOOST_CHECK_CLOSE(results[0], 7.04f, 1e-6);
    BOOST_CHECK_CLOSE(results[1], 7.34f, 1e-6);
    BOOST_CHECK_CLOSE(results[2], 8.54f, 1e-6);
    BOOST_CHECK_CLOSE(results[3], 9.84f, 1e-6);
}

BOOST_AUTO_TEST_CASE(scale_add_vec)
{
    REQUIRES_OPENCL_VERSION(1,2);

    const int N = 10;
    float s = 4.5;
    compute::vector<float> a(N, context);
    compute::vector<float> b(N, context);
    a.assign(N, 1.0f, queue);
    b.assign(N, 2.0f, queue);

    BOOST_COMPUTE_CLOSURE(float, scaleAddVec, (float b, float a), (s),
    {
        return b * s + a;
    });
    compute::transform(b.begin(), b.end(), a.begin(), b.begin(), scaleAddVec, queue);
}

BOOST_AUTO_TEST_SUITE_END()
