//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAccumulate
#include <boost/test/unit_test.hpp>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(sum_int)
{
    int data[] = { 2, 4, 6, 8 };
    boost::compute::vector<int> vector(data, data + 4);
    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 0),
        20
    );

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), -10),
        10
    );

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 5),
        25
    );
}

BOOST_AUTO_TEST_CASE(quotient_int)
{
    int data[] = { 2, 8, 16 };
    boost::compute::vector<int> vector(data, data + 3, context);
    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(
            vector.begin(),
            vector.end(),
            1024,
            boost::compute::divides<int>()
        ),
        4
    );
}

BOOST_AUTO_TEST_CASE(sum_counting_iterator)
{
    // sum 0 -> 9
    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(
            boost::compute::make_counting_iterator(0),
            boost::compute::make_counting_iterator(10),
            0,
            boost::compute::plus<int>(),
            queue
        ),
        45
    );

    // sum 0 -> 9 + 7
    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(
            boost::compute::make_counting_iterator(0),
            boost::compute::make_counting_iterator(10),
            7,
            boost::compute::plus<int>(),
            queue
        ),
        52
    );

    // sum 15 -> 24
    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(
            boost::compute::make_counting_iterator(15),
            boost::compute::make_counting_iterator(25),
            0,
            boost::compute::plus<int>(),
            queue
        ),
        195
    );

    // sum -5 -> 10
    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(
            boost::compute::make_counting_iterator(-5),
            boost::compute::make_counting_iterator(10),
            0,
            boost::compute::plus<int>(),
            queue
        ),
        30
    );

    // sum -5 -> 10 - 2
    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(
            boost::compute::make_counting_iterator(-5),
            boost::compute::make_counting_iterator(10),
            -2,
            boost::compute::plus<int>(),
            queue
        ),
        28
    );
}

BOOST_AUTO_TEST_CASE(sum_iota)
{
    // size 0
    boost::compute::vector<int> vector(0, context);

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 0, queue),
        0
    );

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 4, queue),
        4
    );

    // size 50
    vector.resize(50);
    boost::compute::iota(vector.begin(), vector.end(), 0, queue);

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 0, queue),
        1225
    );

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 11, queue),
        1236
    );

    // size 1000
    vector.resize(1000);
    boost::compute::iota(vector.begin(), vector.end(), 0, queue);

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 0, queue),
        499500
    );

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), -45, queue),
        499455
    );

    // size 1025
    vector.resize(1025);
    boost::compute::iota(vector.begin(), vector.end(), 0, queue);

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 0, queue),
        524800
    );

    BOOST_CHECK_EQUAL(
        boost::compute::accumulate(vector.begin(), vector.end(), 2, queue),
        524802
    );
}

BOOST_AUTO_TEST_CASE(min_and_max)
{
    using boost::compute::int2_;

    int data[] = { 5, 3, 1, 6, 4, 2 };
    boost::compute::vector<int> vector(6, context);
    boost::compute::copy_n(data, 6, vector.begin(), queue);

    BOOST_COMPUTE_FUNCTION(int2_, min_and_max, (int2_, int),
    {
        return (int2)(min(_1.x, _2), max(_1.y, _2));
    });

    int2_ result = boost::compute::accumulate(
        vector.begin(), vector.end(), int2_(100, -100), min_and_max, queue
    );
    BOOST_CHECK_EQUAL(result[0], 1);
    BOOST_CHECK_EQUAL(result[1], 6);
}

BOOST_AUTO_TEST_SUITE_END()
