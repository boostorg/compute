//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestFill
#include <boost/test/unit_test.hpp>

#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/algorithm/fill_n.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(fill_int)
{
    bc::vector<int> vector(1000);
    bc::fill(vector.begin(), vector.end(), 0);
    BOOST_CHECK_EQUAL(vector.front(), 0);
    BOOST_CHECK_EQUAL(vector.back(), 0);

    bc::fill(vector.begin(), vector.end(), 100);
    BOOST_CHECK_EQUAL(vector.front(), 100);
    BOOST_CHECK_EQUAL(vector.back(), 100);

    bc::fill(vector.begin() + 500, vector.end(), 42);
    BOOST_CHECK_EQUAL(vector.front(), 100);
    BOOST_CHECK_EQUAL(vector[499], 100);
    BOOST_CHECK_EQUAL(vector[500], 42);
    BOOST_CHECK_EQUAL(vector.back(), 42);
}

BOOST_AUTO_TEST_CASE(fill_int2)
{
    bc::int2_ value(4, 2);

    bc::vector<bc::int2_> vector(10);
    bc::fill(vector.begin(), vector.end(), value);

    for(size_t i = 0; i < vector.size(); i++){
        bc::int2_ vector_i = vector[i];

        BOOST_CHECK_EQUAL(vector_i[0], 4);
        BOOST_CHECK_EQUAL(vector_i[1], 2);
    }

    value[0] = -2;
    value[1] = -4;
    bc::fill(vector.begin(), vector.end(), value);

    for(size_t i = 0; i < vector.size(); i++){
        bc::int2_ vector_i = vector[i];

        BOOST_CHECK_EQUAL(vector_i[0], -2);
        BOOST_CHECK_EQUAL(vector_i[1], -4);
    }
}

BOOST_AUTO_TEST_CASE(fill_n_float)
{
    bc::vector<float> vector(4);
    bc::fill_n(vector.begin(), 4, 1.5f);
    BOOST_CHECK_EQUAL(vector[0], 1.5f);
    BOOST_CHECK_EQUAL(vector[1], 1.5f);
    BOOST_CHECK_EQUAL(vector[2], 1.5f);
    BOOST_CHECK_EQUAL(vector[3], 1.5f);

    bc::fill_n(vector.begin(), 3, 2.75f);
    BOOST_CHECK_EQUAL(vector[0], 2.75f);
    BOOST_CHECK_EQUAL(vector[1], 2.75f);
    BOOST_CHECK_EQUAL(vector[2], 2.75f);
    BOOST_CHECK_EQUAL(vector[3], 1.5f);

    bc::fill_n(vector.begin() + 1, 2, -3.2f);
    BOOST_CHECK_EQUAL(vector[0], 2.75f);
    BOOST_CHECK_EQUAL(vector[1], -3.2f);
    BOOST_CHECK_EQUAL(vector[2], -3.2f);
    BOOST_CHECK_EQUAL(vector[3], 1.5f);

    bc::fill_n(vector.begin(), 4, 0.0f);
    BOOST_CHECK_EQUAL(vector[0], 0.0f);
    BOOST_CHECK_EQUAL(vector[1], 0.0f);
    BOOST_CHECK_EQUAL(vector[2], 0.0f);
    BOOST_CHECK_EQUAL(vector[3], 0.0f);
}

BOOST_AUTO_TEST_SUITE_END()
