//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestIota
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/permutation_iterator.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(iota_int)
{
    bc::vector<int> vector(4);
    bc::iota(vector.begin(), vector.end(), 0);
    BOOST_CHECK_EQUAL(vector[0], 0);
    BOOST_CHECK_EQUAL(vector[1], 1);
    BOOST_CHECK_EQUAL(vector[2], 2);
    BOOST_CHECK_EQUAL(vector[3], 3);

    bc::iota(vector.begin(), vector.end(), 10);
    BOOST_CHECK_EQUAL(vector[0], 10);
    BOOST_CHECK_EQUAL(vector[1], 11);
    BOOST_CHECK_EQUAL(vector[2], 12);
    BOOST_CHECK_EQUAL(vector[3], 13);

    bc::iota(vector.begin() + 2, vector.end(), -5);
    BOOST_CHECK_EQUAL(vector[0], 10);
    BOOST_CHECK_EQUAL(vector[1], 11);
    BOOST_CHECK_EQUAL(vector[2], -5);
    BOOST_CHECK_EQUAL(vector[3], -4);

    bc::iota(vector.begin(), vector.end() - 2, 4);
    BOOST_CHECK_EQUAL(vector[0], 4);
    BOOST_CHECK_EQUAL(vector[1], 5);
    BOOST_CHECK_EQUAL(vector[2], -5);
    BOOST_CHECK_EQUAL(vector[3], -4);
}

BOOST_AUTO_TEST_CASE(iota_permutation_iterator)
{
    bc::vector<int> output(5);
    bc::fill(output.begin(), output.end(), 0);

    int map_data[] = { 2, 0, 1, 4, 3 };
    bc::vector<int> map(map_data, map_data + 5);

    bc::iota(bc::make_permutation_iterator(output.begin(), map.begin()),
             bc::make_permutation_iterator(output.end(), map.end()),
             1);
    BOOST_CHECK_EQUAL(output[0], 2);
    BOOST_CHECK_EQUAL(output[1], 3);
    BOOST_CHECK_EQUAL(output[2], 1);
    BOOST_CHECK_EQUAL(output[3], 5);
    BOOST_CHECK_EQUAL(output[4], 4);
}

BOOST_AUTO_TEST_SUITE_END()
