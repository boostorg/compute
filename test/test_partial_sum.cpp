//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestPartialSum
#include <boost/test/unit_test.hpp>

#include <vector>
#include <numeric>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/partial_sum.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(partial_sum_int)
{
    int data[] = { 1, 2, 5, 3, 9, 1, 4, 2 };
    bc::vector<int> a(8);
    bc::copy(data, data + 8, a.begin());

    bc::vector<int> b(a.size());
    bc::vector<int>::iterator iter =
        bc::partial_sum(a.begin(), a.end(), b.begin());

    BOOST_CHECK(iter == b.end());
    BOOST_CHECK_EQUAL(b[0], 1);
    BOOST_CHECK_EQUAL(b[1], 3);
    BOOST_CHECK_EQUAL(b[2], 8);
    BOOST_CHECK_EQUAL(b[3], 11);
    BOOST_CHECK_EQUAL(b[4], 20);
    BOOST_CHECK_EQUAL(b[5], 21);
    BOOST_CHECK_EQUAL(b[6], 25);
    BOOST_CHECK_EQUAL(b[7], 27);
}

BOOST_AUTO_TEST_SUITE_END()
