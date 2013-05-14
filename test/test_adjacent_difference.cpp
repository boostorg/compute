//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAdjacentDifference
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/adjacent_difference.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(adjacent_difference_int)
{
    bc::vector<int> a(5, context);
    bc::iota(a.begin(), a.end(), 0);
    CHECK_RANGE_EQUAL(int, 5, a, (0, 1, 2, 3, 4));

    bc::vector<int> b(5, context);
    bc::vector<int>::iterator iter =
        bc::adjacent_difference(a.begin(), a.end(), b.begin());
    BOOST_CHECK(iter == b.end());
    CHECK_RANGE_EQUAL(int, 5, b, (0, 1, 1, 1, 1));

    int data[] = { 1, 9, 36, 48, 81 };
    bc::copy(data, data + 5, a.begin());
    CHECK_RANGE_EQUAL(int, 5, a, (1, 9, 36, 48, 81));

    iter = bc::adjacent_difference(a.begin(), a.end(), b.begin());
    BOOST_CHECK(iter == b.end());
    CHECK_RANGE_EQUAL(int, 5, b, (1, 8, 27, 12, 33));
}

BOOST_AUTO_TEST_SUITE_END()
