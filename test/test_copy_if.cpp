//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestCopyIf
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/algorithm/copy_if.hpp>
#include <boost/compute/container/vector.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(copy_if_int)
{
    int data[] = { 1, 6, 3, 5, 8, 2, 4 };
    bc::vector<int> input(data, data + 7);

    bc::vector<int> output(input.size());
    bc::fill(output.begin(), output.end(), -1);

    using ::boost::compute::_1;

    bc::vector<int>::iterator iter =
        bc::copy_if(input.begin(), input.end(), output.begin(), _1 < 5);
    BOOST_VERIFY(iter == output.begin() + 4);
    BOOST_CHECK_EQUAL(output[0], 1);
    BOOST_CHECK_EQUAL(output[1], 3);
    BOOST_CHECK_EQUAL(output[2], 2);
    BOOST_CHECK_EQUAL(output[3], 4);
    BOOST_CHECK_EQUAL(output[4], -1);
    BOOST_CHECK_EQUAL(output[5], -1);
    BOOST_CHECK_EQUAL(output[6], -1);

    bc::fill(output.begin(), output.end(), 42);
    iter =
        bc::copy_if(input.begin(), input.end(), output.begin(), _1 * 2 >= 10);
    BOOST_VERIFY(iter == output.begin() + 3);
    BOOST_CHECK_EQUAL(output[0], 6);
    BOOST_CHECK_EQUAL(output[1], 5);
    BOOST_CHECK_EQUAL(output[2], 8);
    BOOST_CHECK_EQUAL(output[3], 42);
    BOOST_CHECK_EQUAL(output[4], 42);
    BOOST_CHECK_EQUAL(output[5], 42);
    BOOST_CHECK_EQUAL(output[6], 42);
}

BOOST_AUTO_TEST_CASE(copy_if_odd)
{
    int data[] = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
    bc::vector<int> input(data, data + 10);

    using ::boost::compute::_1;

    bc::vector<int> odds(input.size());
    bc::vector<int>::iterator odds_end =
        bc::copy_if(input.begin(), input.end(), odds.begin(), _1 % 2 == 1);
    BOOST_CHECK(odds_end == odds.begin() + 6);
    BOOST_CHECK_EQUAL(int(odds[0]), int(1));
    BOOST_CHECK_EQUAL(int(odds[1]), int(3));
    BOOST_CHECK_EQUAL(int(odds[2]), int(5));
    BOOST_CHECK_EQUAL(int(odds[3]), int(1));
    BOOST_CHECK_EQUAL(int(odds[4]), int(3));
    BOOST_CHECK_EQUAL(int(odds[5]), int(5));

    bc::vector<int> evens(input.size());
    bc::vector<int>::iterator evens_end =
        bc::copy_if(input.begin(), input.end(), evens.begin(), _1 % 2 == 0);
    BOOST_CHECK(evens_end == evens.begin() + 4);
    BOOST_CHECK_EQUAL(int(evens[0]), int(2));
    BOOST_CHECK_EQUAL(int(evens[1]), int(4));
    BOOST_CHECK_EQUAL(int(evens[2]), int(2));
    BOOST_CHECK_EQUAL(int(evens[3]), int(4));
}

BOOST_AUTO_TEST_CASE(clip_points_below_plane)
{
    float data[] = { 1.0f, 2.0f, 3.0f, 0.0f,
                     -1.0f, 2.0f, 3.0f, 0.0f,
                     -2.0f, -3.0f, 4.0f, 0.0f,
                     4.0f, -3.0f, 2.0f, 0.0f };
    bc::vector<bc::float4_> points(reinterpret_cast<bc::float4_ *>(data),
                                   reinterpret_cast<bc::float4_ *>(data) + 4);

    // create output vector filled with (0, 0, 0, 0)
    bc::vector<bc::float4_> output(points.size());
    bc::fill(output.begin(), output.end(), bc::float4_(0.0f, 0.0f, 0.0f, 0.0f));

    // define the plane (at origin, +X normal)
    bc::float4_ plane_origin(0.0f, 0.0f, 0.0f, 0.0f);
    bc::float4_ plane_normal(1.0f, 0.0f, 0.0f, 0.0f);

    using ::boost::compute::_1;
    using ::boost::compute::lambda::dot;

    bc::vector<bc::float4_>::const_iterator iter =
        bc::copy_if(points.begin(),
                    points.end(),
                    output.begin(),
                    dot(_1 - plane_origin, plane_normal) > 0.0f);
    BOOST_CHECK(iter == output.begin() + 2);
}
