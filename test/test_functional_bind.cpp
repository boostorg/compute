//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestFunctionalBind
#include <boost/test/unit_test.hpp>

#include <boost/compute/function.hpp>
#include <boost/compute/algorithm/copy_n.hpp>
#include <boost/compute/algorithm/count_if.hpp>
#include <boost/compute/algorithm/find_if.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/bind.hpp>
#include <boost/compute/functional/common.hpp>
#include <boost/compute/functional/operator.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace compute = boost::compute;

using compute::placeholders::_1;
using compute::placeholders::_2;

BOOST_AUTO_TEST_CASE(transform_plus_two)
{
    int data[] = { 1, 2, 3, 4 };
    compute::vector<int> vector(4, context);
    compute::copy_n(data, 4, vector.begin(), queue);

    compute::transform(
        vector.begin(), vector.end(), vector.begin(),
        compute::bind(compute::plus<int>(), _1, 2),
        queue
    );

    CHECK_RANGE_EQUAL(int, 4, vector, (3, 4, 5, 6));
}

BOOST_AUTO_TEST_CASE(transform_pow_two)
{
    float data[] = { 2, 3, 4, 5 };
    compute::vector<float> vector(4, context);
    compute::copy_n(data, 4, vector.begin(), queue);

    compute::transform(
        vector.begin(), vector.end(), vector.begin(),
        compute::bind(compute::pow<float>(), 2.0f, _1),
        queue
    );
    CHECK_RANGE_EQUAL(float, 4, vector, (4.0f, 8.0f, 16.0f, 32.0f));
}

BOOST_AUTO_TEST_CASE(find_if_equal)
{
    int data[] = { 1, 2, 3, 4 };
    compute::vector<int> vector(4, context);
    compute::copy_n(data, 4, vector.begin(), queue);

    BOOST_CHECK(
        compute::find_if(
            vector.begin(), vector.end(),
            compute::bind(compute::equal_to<int>(), _1, 3),
            queue
        ) == vector.begin() + 2
    );
}

BOOST_AUTO_TEST_CASE(compare_less_than)
{
    int data[] = { 1, 2, 3, 4 };
    compute::vector<int> vector(data, data + 4, queue);

    int count = boost::compute::count_if(
        vector.begin(), vector.end(),
        compute::bind(compute::less<int>(), _1, 3),
        queue
    );
    BOOST_CHECK_EQUAL(count, 2);

    count = boost::compute::count_if(
        vector.begin(), vector.end(),
        compute::bind(compute::less<int>(), 3, _1),
        queue
    );
    BOOST_CHECK_EQUAL(count, 1);
}

BOOST_AUTO_TEST_CASE(subtract_ranges)
{
    int data1[] = { 1, 2, 3, 4 };
    int data2[] = { 4, 3, 2, 1 };

    compute::vector<int> vector1(data1, data1 + 4, queue);
    compute::vector<int> vector2(data2, data2 + 4, queue);

    compute::vector<int> result(4, context);

    compute::transform(
        vector1.begin(),
        vector1.end(),
        vector2.begin(),
        result.begin(),
        compute::bind(compute::minus<int>(), _1, _2),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, result, (-3, -1, 1, 3));

    compute::transform(
        vector1.begin(),
        vector1.end(),
        vector2.begin(),
        result.begin(),
        compute::bind(compute::minus<int>(), _2, _1),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, result, (3, 1, -1, -3));

    compute::transform(
        vector1.begin(),
        vector1.end(),
        vector2.begin(),
        result.begin(),
        compute::bind(compute::minus<int>(), 5, _1),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, result, (4, 3, 2, 1));

    compute::transform(
        vector1.begin(),
        vector1.end(),
        vector2.begin(),
        result.begin(),
        compute::bind(compute::minus<int>(), 5, _2),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, result, (1, 2, 3, 4));
}

BOOST_AUTO_TEST_CASE(clamp_values)
{
    int data[] = { 1, 2, 3, 4 };
    compute::vector<int> vector(data, data + 4, queue);

    compute::transform(
        vector.begin(), vector.end(), vector.begin(),
        compute::bind(compute::clamp<int>(), _1, 2, 3),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, vector, (2, 2, 3, 3));
}

BOOST_AUTO_TEST_CASE(bind_custom_function)
{
    int data[] = { 1, 2, 3, 4 };
    compute::vector<int> vector(data, data + 4, queue);

    BOOST_COMPUTE_FUNCTION(int, x_if_odd_else_y, (int x, int y),
    {
        if(x & 1)
            return x;
        else
            return y;
    });

    compute::transform(
        vector.begin(), vector.end(), vector.begin(),
        compute::bind(x_if_odd_else_y, _1, 9),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, vector, (1, 9, 3, 9));

    compute::copy(
        data, data + 4, vector.begin(), queue
    );

    compute::transform(
        vector.begin(), vector.end(), vector.begin(),
        compute::bind(x_if_odd_else_y, 2, _1),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, vector, (1, 2, 3, 4));
}

BOOST_AUTO_TEST_SUITE_END()
