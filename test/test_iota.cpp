//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestIota
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/permutation_iterator.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(iota_int)
{
    bc::vector<int> vector(4);
    bc::iota(vector.begin(), vector.end(), 0);
    CHECK_RANGE_EQUAL(int, 4, vector, (0, 1, 2, 3));

    bc::iota(vector.begin(), vector.end(), 10);
    CHECK_RANGE_EQUAL(int, 4, vector, (10, 11, 12, 13));

    bc::iota(vector.begin() + 2, vector.end(), -5);
    CHECK_RANGE_EQUAL(int, 4, vector, (10, 11, -5, -4));

    bc::iota(vector.begin(), vector.end() - 2, 4);
    CHECK_RANGE_EQUAL(int, 4, vector, (4, 5, -5, -4));
}

BOOST_AUTO_TEST_CASE(iota_doctest)
{
    boost::compute::vector<int> vec(3, context);

//! [iota]
boost::compute::iota(vec.begin(), vec.end(), 0, queue);
//! [iota]

    CHECK_RANGE_EQUAL(int, 3, vec, (0, 1, 2));
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
    CHECK_RANGE_EQUAL(int, 5, output, (2, 3, 1, 5, 4));
}

BOOST_AUTO_TEST_SUITE_END()
