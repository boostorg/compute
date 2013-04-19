//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestReverse
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/algorithm/reverse.hpp>
#include <boost/compute/algorithm/reverse_copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(reverse_int)
{
    bc::vector<int> vector(5);
    bc::iota(vector.begin(), vector.end(), 0);
    BOOST_CHECK_EQUAL(vector[0], 0);
    BOOST_CHECK_EQUAL(vector[1], 1);
    BOOST_CHECK_EQUAL(vector[2], 2);
    BOOST_CHECK_EQUAL(vector[3], 3);
    BOOST_CHECK_EQUAL(vector[4], 4);

    bc::reverse(vector.begin(), vector.end());
    BOOST_CHECK_EQUAL(vector[0], 4);
    BOOST_CHECK_EQUAL(vector[1], 3);
    BOOST_CHECK_EQUAL(vector[2], 2);
    BOOST_CHECK_EQUAL(vector[3], 1);
    BOOST_CHECK_EQUAL(vector[4], 0);

    bc::reverse(vector.begin() + 1, vector.end());
    BOOST_CHECK_EQUAL(vector[0], 4);
    BOOST_CHECK_EQUAL(vector[1], 0);
    BOOST_CHECK_EQUAL(vector[2], 1);
    BOOST_CHECK_EQUAL(vector[3], 2);
    BOOST_CHECK_EQUAL(vector[4], 3);

    bc::reverse(vector.begin() + 1, vector.end() - 1);
    BOOST_CHECK_EQUAL(vector[0], 4);
    BOOST_CHECK_EQUAL(vector[1], 2);
    BOOST_CHECK_EQUAL(vector[2], 1);
    BOOST_CHECK_EQUAL(vector[3], 0);
    BOOST_CHECK_EQUAL(vector[4], 3);

    bc::reverse(vector.begin(), vector.end() - 2);
    BOOST_CHECK_EQUAL(vector[0], 1);
    BOOST_CHECK_EQUAL(vector[1], 2);
    BOOST_CHECK_EQUAL(vector[2], 4);
    BOOST_CHECK_EQUAL(vector[3], 0);
    BOOST_CHECK_EQUAL(vector[4], 3);

    vector.resize(6);
    bc::iota(vector.begin(), vector.end(), 10);
    BOOST_CHECK_EQUAL(vector[0], 10);
    BOOST_CHECK_EQUAL(vector[1], 11);
    BOOST_CHECK_EQUAL(vector[2], 12);
    BOOST_CHECK_EQUAL(vector[3], 13);
    BOOST_CHECK_EQUAL(vector[4], 14);
    BOOST_CHECK_EQUAL(vector[5], 15);

    bc::reverse(vector.begin(), vector.end());
    BOOST_CHECK_EQUAL(vector[0], 15);
    BOOST_CHECK_EQUAL(vector[1], 14);
    BOOST_CHECK_EQUAL(vector[2], 13);
    BOOST_CHECK_EQUAL(vector[3], 12);
    BOOST_CHECK_EQUAL(vector[4], 11);
    BOOST_CHECK_EQUAL(vector[5], 10);

    bc::reverse(vector.begin() + 3, vector.end());
    BOOST_CHECK_EQUAL(vector[0], 15);
    BOOST_CHECK_EQUAL(vector[1], 14);
    BOOST_CHECK_EQUAL(vector[2], 13);
    BOOST_CHECK_EQUAL(vector[3], 10);
    BOOST_CHECK_EQUAL(vector[4], 11);
    BOOST_CHECK_EQUAL(vector[5], 12);

    bc::reverse(vector.begin() + 1, vector.end() - 2);
    BOOST_CHECK_EQUAL(vector[0], 15);
    BOOST_CHECK_EQUAL(vector[1], 10);
    BOOST_CHECK_EQUAL(vector[2], 13);
    BOOST_CHECK_EQUAL(vector[3], 14);
    BOOST_CHECK_EQUAL(vector[4], 11);
    BOOST_CHECK_EQUAL(vector[5], 12);
}

BOOST_AUTO_TEST_CASE(reverse_copy_int)
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
        bc::reverse_copy(a.begin(), a.end(), b.begin());
    BOOST_CHECK(iter == b.end());
    BOOST_CHECK_EQUAL(b[0], 4);
    BOOST_CHECK_EQUAL(b[1], 3);
    BOOST_CHECK_EQUAL(b[2], 2);
    BOOST_CHECK_EQUAL(b[3], 1);
    BOOST_CHECK_EQUAL(b[4], 0);
}

BOOST_AUTO_TEST_CASE(reverse_copy_counting_iterator)
{
    bc::vector<int> vector(5);
    bc::reverse_copy(
        bc::make_counting_iterator(1),
        bc::make_counting_iterator(6),
        vector.begin()
    );

    BOOST_CHECK_EQUAL(vector[0], 5);
    BOOST_CHECK_EQUAL(vector[1], 4);
    BOOST_CHECK_EQUAL(vector[2], 3);
    BOOST_CHECK_EQUAL(vector[3], 2);
    BOOST_CHECK_EQUAL(vector[4], 1);
}

BOOST_AUTO_TEST_SUITE_END()
