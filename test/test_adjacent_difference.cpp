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

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(adjacent_difference_int)
{
    bc::vector<int> a(5, context);
    bc::iota(a.begin(), a.end(), 0);
    BOOST_CHECK_EQUAL(a[0], 0);
    BOOST_CHECK_EQUAL(a[1], 1);
    BOOST_CHECK_EQUAL(a[2], 2);
    BOOST_CHECK_EQUAL(a[3], 3);
    BOOST_CHECK_EQUAL(a[4], 4);

    bc::vector<int> b(5, context);
    bc::vector<int>::iterator iter =
        bc::adjacent_difference(a.begin(), a.end(), b.begin());
    BOOST_CHECK(iter == b.end());
    BOOST_CHECK_EQUAL(b[0], 0);
    BOOST_CHECK_EQUAL(b[1], 1);
    BOOST_CHECK_EQUAL(b[2], 1);
    BOOST_CHECK_EQUAL(b[3], 1);
    BOOST_CHECK_EQUAL(b[4], 1);

    int data[] = { 1, 9, 36, 48, 81 };
    bc::copy(data, data + 5, a.begin());
    BOOST_CHECK_EQUAL(a[0], 1);
    BOOST_CHECK_EQUAL(a[1], 9);
    BOOST_CHECK_EQUAL(a[2], 36);
    BOOST_CHECK_EQUAL(a[3], 48);
    BOOST_CHECK_EQUAL(a[4], 81);

    iter = bc::adjacent_difference(a.begin(), a.end(), b.begin());
    BOOST_CHECK(iter == b.end());
    BOOST_CHECK_EQUAL(b[0], 1);
    BOOST_CHECK_EQUAL(b[1], 8);
    BOOST_CHECK_EQUAL(b[2], 27);
    BOOST_CHECK_EQUAL(b[3], 12);
    BOOST_CHECK_EQUAL(b[4], 33);
}

BOOST_AUTO_TEST_SUITE_END()
