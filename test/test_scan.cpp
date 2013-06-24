//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestScan
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(inclusive_scan_int)
{
    int data[] = { 1, 2, 1, 2, 3 };
    bc::vector<int> vector(data, data + 5, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(5));

    bc::vector<int> result(5, context);
    BOOST_CHECK_EQUAL(result.size(), size_t(5));

    // inclusive scan
    bc::inclusive_scan(vector.begin(), vector.end(), result.begin());
    CHECK_RANGE_EQUAL(int, 5, result, (1, 3, 4, 6, 9));

    // in-place inclusive scan
    CHECK_RANGE_EQUAL(int, 5, vector, (1, 2, 1, 2, 3));
    bc::inclusive_scan(vector.begin(), vector.end(), vector.begin());
    CHECK_RANGE_EQUAL(int, 5, vector, (1, 3, 4, 6, 9));
}

BOOST_AUTO_TEST_CASE(exclusive_scan_int)
{
    int data[] = { 1, 2, 1, 2, 3 };
    bc::vector<int> vector(data, data + 5, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(5));

    bc::vector<int> result(5, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(5));

    // exclusive scan
    bc::exclusive_scan(vector.begin(), vector.end(), result.begin());
    CHECK_RANGE_EQUAL(int, 5, result, (0, 1, 3, 4, 6));

    // in-place exclusive scan
    CHECK_RANGE_EQUAL(int, 5, vector, (1, 2, 1, 2, 3));
    bc::exclusive_scan(vector.begin(), vector.end(), vector.begin());
    CHECK_RANGE_EQUAL(int, 5, vector, (0, 1, 3, 4, 6));
}

BOOST_AUTO_TEST_CASE(inclusive_scan_int2)
{
    using boost::compute::int2_;

    int data[] = { 1, 2,
                   3, 4,
                   5, 6,
                   7, 8,
                   9, 0 };

    boost::compute::vector<int2_> input(reinterpret_cast<int2_*>(data),
                                        reinterpret_cast<int2_*>(data) + 5);
    BOOST_CHECK_EQUAL(input.size(), size_t(5));

    boost::compute::vector<int2_> output(5);
    boost::compute::inclusive_scan(input.begin(), input.end(), output.begin());
    CHECK_RANGE_EQUAL(
        int2_, 5, output,
        (int2_(1, 2), int2_(4, 6), int2_(9, 12), int2_(16, 20), int2_(25, 20))
    );
}

BOOST_AUTO_TEST_CASE(inclusive_scan_counting_iterator)
{
    bc::vector<int> result(10, context);
    bc::inclusive_scan(bc::make_counting_iterator(1),
                       bc::make_counting_iterator(11),
                       result.begin());
    CHECK_RANGE_EQUAL(int, 10, result, (1, 3, 6, 10, 15, 21, 28, 36, 45, 55));
}

BOOST_AUTO_TEST_CASE(exclusive_scan_counting_iterator)
{
    bc::vector<int> result(10, context);
    bc::exclusive_scan(bc::make_counting_iterator(1),
                       bc::make_counting_iterator(11),
                       result.begin());
    CHECK_RANGE_EQUAL(int, 10, result, (0, 1, 3, 6, 10, 15, 21, 28, 36, 45));
}

BOOST_AUTO_TEST_CASE(inclusive_scan_transform_iterator)
{
    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    bc::vector<float> input(data, data + 5, context);
    bc::vector<float> output(5, context);

    // normal inclusive scan of the input
    bc::inclusive_scan(input.begin(), input.end(), output.begin());
    bc::system::finish();
    BOOST_CHECK_CLOSE(float(output[0]), 1.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[1]), 3.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[2]), 6.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[3]), 10.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[4]), 15.0f, 1e-4f);

    // inclusive scan of squares of the input
    using ::boost::compute::_1;

    bc::inclusive_scan(bc::make_transform_iterator(input.begin(), pown(_1, 2)),
                       bc::make_transform_iterator(input.end(), pown(_1, 2)),
                       output.begin());
    bc::system::finish();
    BOOST_CHECK_CLOSE(float(output[0]), 1.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[1]), 5.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[2]), 14.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[3]), 30.0f, 1e-4f);
    BOOST_CHECK_CLOSE(float(output[4]), 55.0f, 1e-4f);
}

BOOST_AUTO_TEST_SUITE_END()
