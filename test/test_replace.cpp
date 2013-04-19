//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestReplace
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/algorithm/replace.hpp>
#include <boost/compute/algorithm/replace_copy.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(replace_int)
{
    bc::vector<int> vector(5);
    bc::iota(vector.begin(), vector.end(), 0);
    BOOST_CHECK_EQUAL(vector[0], 0);
    BOOST_CHECK_EQUAL(vector[1], 1);
    BOOST_CHECK_EQUAL(vector[2], 2);
    BOOST_CHECK_EQUAL(vector[3], 3);
    BOOST_CHECK_EQUAL(vector[4], 4);

    bc::replace(vector.begin(), vector.end(), 2, 6);
    BOOST_CHECK_EQUAL(vector[0], 0);
    BOOST_CHECK_EQUAL(vector[1], 1);
    BOOST_CHECK_EQUAL(vector[2], 6);
    BOOST_CHECK_EQUAL(vector[3], 3);
    BOOST_CHECK_EQUAL(vector[4], 4);
}

BOOST_AUTO_TEST_CASE(replace_copy_int)
{
    bc::vector<int> a(5);
    bc::iota(a.begin(), a.end(), 0);
    BOOST_CHECK_EQUAL(a[0], 0);
    BOOST_CHECK_EQUAL(a[1], 1);
    BOOST_CHECK_EQUAL(a[2], 2);
    BOOST_CHECK_EQUAL(a[3], 3);
    BOOST_CHECK_EQUAL(a[4], 4);

    bc::vector<int> b(5);
    bc::vector<int>::iterator iter =
        bc::replace_copy(a.begin(), a.end(), b.begin(), 3, 9);
    BOOST_CHECK(iter == b.end());
    BOOST_CHECK_EQUAL(b[0], 0);
    BOOST_CHECK_EQUAL(b[1], 1);
    BOOST_CHECK_EQUAL(b[2], 2);
    BOOST_CHECK_EQUAL(b[3], 9);
    BOOST_CHECK_EQUAL(b[4], 4);

    // ensure 'a' was not modified
    BOOST_CHECK_EQUAL(a[0], 0);
    BOOST_CHECK_EQUAL(a[1], 1);
    BOOST_CHECK_EQUAL(a[2], 2);
    BOOST_CHECK_EQUAL(a[3], 3);
    BOOST_CHECK_EQUAL(a[4], 4);
}

BOOST_AUTO_TEST_SUITE_END()
