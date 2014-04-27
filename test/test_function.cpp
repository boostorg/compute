//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestFunction
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/types/pair.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(add_three)
{
    BOOST_COMPUTE_FUNCTION(int, add_three, (int x),
    {
        return x + 3;
    });

    int data[] = { 1, 2, 3, 4 };
    compute::vector<int> vector(data, data + 4, queue);

    compute::transform(
        vector.begin(), vector.end(), vector.begin(), add_three, queue
    );
    CHECK_RANGE_EQUAL(int, 4, vector, (4, 5, 6, 7));
}

BOOST_AUTO_TEST_CASE(sum_odd_values)
{
    BOOST_COMPUTE_FUNCTION(int, add_odd_value, (int sum, int value),
    {
        if(value & 1){
            return sum + value;
        }
        else {
            return sum + 0;
        }
    });

    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    compute::vector<int> vector(data, data + 8, queue);

    int result = compute::accumulate(
        vector.begin(), vector.end(), 0, add_odd_value, queue
    );
    BOOST_CHECK_EQUAL(result, 16);
}

BOOST_AUTO_TEST_CASE(sort_pairs)
{
    std::vector<std::pair<int, float> > data;
    data.push_back(std::make_pair(1, 2.3f));
    data.push_back(std::make_pair(0, 4.2f));
    data.push_back(std::make_pair(2, 1.0f));

    compute::vector<std::pair<int, float> > vector(data.size());
    compute::copy(data.begin(), data.end(), vector.begin(), queue);

    // sort by first component
    BOOST_COMPUTE_FUNCTION(bool, compare_first, (std::pair<int, float> a, std::pair<int, float> b),
    {
        return a.first < b.first;
    });

    compute::sort(vector.begin(), vector.end(), compare_first, queue);
    compute::copy(vector.begin(), vector.end(), data.begin(), queue);
    BOOST_CHECK(data[0] == std::make_pair(0, 4.2f));
    BOOST_CHECK(data[1] == std::make_pair(1, 2.3f));
    BOOST_CHECK(data[2] == std::make_pair(2, 1.0f));

    // sort by second component
    BOOST_COMPUTE_FUNCTION(bool, compare_second, (std::pair<int, float> a, std::pair<int, float> b),
    {
        return a.second < b.second;
    });

    compute::sort(vector.begin(), vector.end(), compare_second, queue);
    compute::copy(vector.begin(), vector.end(), data.begin(), queue);
    BOOST_CHECK(data[0] == std::make_pair(2, 1.0f));
    BOOST_CHECK(data[1] == std::make_pair(1, 2.3f));
    BOOST_CHECK(data[2] == std::make_pair(0, 4.2f));
}

BOOST_AUTO_TEST_CASE(transform_zip_iterator)
{
    float float_data[] = { 1.f, 2.f, 3.f, 4.f };
    compute::vector<float> input_floats(float_data, float_data + 4, queue);

    int int_data[] = { 2, 4, 6, 8 };
    compute::vector<int> input_ints(int_data, int_data + 4, queue);

    compute::vector<float> results(4, context);

    BOOST_COMPUTE_FUNCTION(float, tuple_pown, (boost::tuple<float, int> x),
    {
        return pown(boost_tuple_get(x, 0), boost_tuple_get(x, 1));
    });

    compute::transform(
        compute::make_zip_iterator(
            boost::make_tuple(input_floats.begin(), input_ints.begin())
        ),
        compute::make_zip_iterator(
            boost::make_tuple(input_floats.end(), input_ints.end())
        ),
        results.begin(),
        tuple_pown,
        queue
    );

    float results_data[4];
    compute::copy(results.begin(), results.end(), results_data, queue);
    BOOST_CHECK_CLOSE(results_data[0], 1.f, 1e-4);
    BOOST_CHECK_CLOSE(results_data[1], 16.f, 1e-4);
    BOOST_CHECK_CLOSE(results_data[2], 729.f, 1e-4);
    BOOST_CHECK_CLOSE(results_data[3], 65536.f, 1e-4);
}

static BOOST_COMPUTE_FUNCTION(int, static_function, (int x),
{
    return x + 5;
});

BOOST_AUTO_TEST_CASE(test_static_function)
{
    int data[] = { 1, 2, 3, 4};
    compute::vector<int> vec(data, data + 4, queue);

    compute::transform(
        vec.begin(), vec.end(), vec.begin(), static_function, queue
    );
    CHECK_RANGE_EQUAL(int, 4, vec, (6, 7, 8, 9));
}

template<class T>
inline compute::function<T(T)> make_negate_function()
{
    BOOST_COMPUTE_FUNCTION(T, negate, (const T x),
    {
        return -x;
    });

    return negate;
}

BOOST_AUTO_TEST_CASE(test_templated_function)
{
    int int_data[] = { 1, 2, 3, 4 };
    compute::vector<int> int_vec(int_data, int_data + 4, queue);

    compute::function<int(int)> negate_int = make_negate_function<int>();
    compute::transform(
        int_vec.begin(), int_vec.end(), int_vec.begin(), negate_int, queue
    );
    CHECK_RANGE_EQUAL(int, 4, int_vec, (-1, -2, -3, -4));

    float float_data[] = { 1.1f, 2.2f, 3.3f, 4.4f };
    compute::vector<float> float_vec(float_data, float_data + 4, queue);

    compute::function<float(float)> negate_float = make_negate_function<float>();
    compute::transform(
        float_vec.begin(), float_vec.end(), float_vec.begin(), negate_float, queue
    );
    CHECK_RANGE_EQUAL(float, 4, float_vec, (-1.1f, -2.2f, -3.3f, -4.4f));
}

BOOST_AUTO_TEST_SUITE_END()
