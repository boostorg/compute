//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestReduce
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/constant_iterator.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(reduce_int)
{
    int data[] = { 1, 5, 9, 13, 17 };
    compute::vector<int> vector(data, data + 5, context);
    int sum;
    compute::reduce(vector.begin(), vector.end(), &sum, 0, compute::plus<int>(), queue);
    BOOST_CHECK_EQUAL(sum, 45);

    int product;
    compute::reduce(vector.begin(), vector.end(), &product, 1, compute::multiplies<int>(), queue);
    BOOST_CHECK_EQUAL(product, 9945);
}

BOOST_AUTO_TEST_CASE(reduce_on_device)
{
    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    compute::vector<int> input(data, data + 8, context);
    compute::vector<int> result(2, context);
    compute::reduce(input.begin(), input.begin() + 4, result.begin(), 0, compute::plus<int>(), queue);
    compute::reduce(input.begin() + 4, input.end(), result.end() - 1, 0, compute::plus<int>(), queue);
    CHECK_RANGE_EQUAL(int, 2, result, (10, 26));
}

BOOST_AUTO_TEST_CASE(reduce_int_min_max)
{
    int data[] = { 11, 5, 92, 13, 42 };
    compute::vector<int> vector(data, data + 5, context);
    int min_value;
    compute::reduce(
        vector.begin(),
        vector.end(),
        &min_value,
        (std::numeric_limits<int>::max)(),
        compute::min<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(min_value, 5);

    int max_value;
    compute::reduce(
        vector.begin(),
        vector.end(),
        &max_value,
        (std::numeric_limits<int>::min)(),
        compute::max<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(max_value, 92);
}

BOOST_AUTO_TEST_CASE(reduce_int2)
{
    std::vector<compute::int2_> data;
    for(int i = 0; i < 6; i++){
        compute::int2_ value;
        value[0] = i + 1;
        value[1] = 2 * i + 1;
        data.push_back(value);
    }

    compute::vector<compute::int2_> vector(data.begin(), data.end(), context);

    compute::int2_ sum;
    compute::reduce(
        vector.begin(),
        vector.end(),
        &sum,
        compute::int2_(0, 0),
        compute::plus<compute::int2_>(),
        queue
    );
    BOOST_CHECK_EQUAL(sum, compute::int2_(21, 36));
}

BOOST_AUTO_TEST_CASE(reduce_pinned_vector)
{
    int data[] = { 2, 5, 8, 11, 15 };
    std::vector<int> vector(data, data + 5);

    compute::buffer buffer(context,
                           vector.size() * sizeof(int),
                           compute::buffer::read_only | compute::buffer::use_host_ptr,
                           &vector[0]);

    int sum;
    compute::reduce(
        compute::make_buffer_iterator<int>(buffer, 0),
        compute::make_buffer_iterator<int>(buffer, 5),
        &sum,
        0,
        compute::plus<int>()
    );
    BOOST_CHECK_EQUAL(sum, 41);
}

BOOST_AUTO_TEST_CASE(reduce_constant_iterator)
{
    int result;
    compute::reduce(
        compute::make_constant_iterator(1, 0),
        compute::make_constant_iterator(1, 5),
        &result,
        0,
        compute::plus<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(result, 5);

    compute::reduce(
        compute::make_constant_iterator(3, 0),
        compute::make_constant_iterator(3, 5),
        &result,
        0,
        compute::plus<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(result, 15);

    compute::reduce(
        compute::make_constant_iterator(2, 0),
        compute::make_constant_iterator(2, 5),
        &result,
        1,
        compute::multiplies<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(result, 32);
}

BOOST_AUTO_TEST_CASE(reduce_counting_iterator)
{
    int result;
    compute::reduce(
        compute::make_counting_iterator(1),
        compute::make_counting_iterator(10),
        &result,
        0,
        compute::plus<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(result, 45);

    compute::reduce(
        compute::make_counting_iterator(1),
        compute::make_counting_iterator(5),
        &result,
        1,
        compute::multiplies<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(result, 24);
}

BOOST_AUTO_TEST_CASE(reduce_transform_iterator)
{
    using ::boost::compute::_1;

    int data[] = { 1, 3, 5, 7, 9 };
    compute::vector<int> vector(data, data + 5, context);

    int sum;
    compute::reduce(
        compute::make_transform_iterator(vector.begin(), _1 + 1),
        compute::make_transform_iterator(vector.end(), _1 + 1),
        &sum,
        0,
        compute::plus<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(sum, 30);

    compute::reduce(
        compute::make_transform_iterator(vector.begin(), _1 > 4),
        compute::make_transform_iterator(vector.end(), _1 > 4),
        &sum,
        0,
        compute::plus<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(sum, 3);

    compute::reduce(
        compute::make_transform_iterator(vector.begin(), _1 * _1),
        compute::make_transform_iterator(vector.end(), _1 * _1),
        &sum,
        0,
        compute::plus<int>(),
        queue
    );
    BOOST_CHECK_EQUAL(sum, 165);
}

BOOST_AUTO_TEST_CASE(min_and_max)
{
    using compute::int2_;

    int data[] = { 5, 3, 1, 6, 4, 2 };
    compute::vector<int> vector(6, context);
    compute::copy_n(data, 6, vector.begin(), queue);

    compute::function<int2_(int2_, int)> min_and_max =
        compute::make_function_from_source<int2_(int2_, int)>("min_and_max",
            "int2 min_and_max(int2 r, int x) { return (int2)(min(r.x, x), max(r.y, x)); }"
        );

    int2_ result(std::numeric_limits<int>::max(), std::numeric_limits<int>::min());
    compute::reduce(vector.begin(), vector.end(), &result, result, min_and_max, queue);
    BOOST_CHECK_EQUAL(result[0], 1);
    BOOST_CHECK_EQUAL(result[1], 6);
}

BOOST_AUTO_TEST_SUITE_END()
