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
    BOOST_COMPUTE_FUNCTION(int, add_three, (int),
    {
        return _1 + 3;
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
    BOOST_COMPUTE_FUNCTION(int, add_odd_value, (int, int),
    {
        if(_2 & 1){
            return _1 + _2;
        }
        else {
            return _1 + 0;
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
    BOOST_COMPUTE_FUNCTION(bool, compare_first, (std::pair<int, float>, std::pair<int, float>),
    {
        return _1.first < _2.first;
    });

    compute::sort(vector.begin(), vector.end(), compare_first, queue);
    compute::copy(vector.begin(), vector.end(), data.begin(), queue);
    BOOST_CHECK(data[0] == std::make_pair(0, 4.2f));
    BOOST_CHECK(data[1] == std::make_pair(1, 2.3f));
    BOOST_CHECK(data[2] == std::make_pair(2, 1.0f));

    // sort by second component
    BOOST_COMPUTE_FUNCTION(bool, compare_second, (std::pair<int, float>, std::pair<int, float>),
    {
        return _1.second < _2.second;
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

    BOOST_COMPUTE_FUNCTION(float, tuple_pown, (boost::tuple<float, int>),
    {
        return pown(boost_tuple_get(_1, 0), boost_tuple_get(_1, 1));
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

BOOST_AUTO_TEST_SUITE_END()
