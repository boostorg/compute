//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestFind
#include <boost/test/unit_test.hpp>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/algorithm/find_if.hpp>
#include <boost/compute/algorithm/find_if_not.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/constant_buffer_iterator.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(find_int)
{
    int data[] = { 9, 15, 1, 4, 9, 9, 4, 15, 12, 1 };
    bc::vector<int> vector(data, data + 10);

    bc::vector<int>::iterator iter =
        bc::find(vector.begin(), vector.end(), 4);
    BOOST_CHECK(iter == vector.begin() + 3);
    BOOST_CHECK_EQUAL(*iter, 4);

    iter = bc::find(vector.begin(), vector.end(), 12);
    BOOST_CHECK(iter == vector.begin() + 8);
    BOOST_CHECK_EQUAL(*iter, 12);

    iter = bc::find(vector.begin(), vector.end(), 1);
    BOOST_CHECK(iter == vector.begin() + 2);
    BOOST_CHECK_EQUAL(*iter, 1);

    iter = bc::find(vector.begin(), vector.end(), 9);
    BOOST_CHECK(iter == vector.begin());
    BOOST_CHECK_EQUAL(*iter, 9);

    iter = bc::find(vector.begin(), vector.end(), 100);
    BOOST_CHECK(iter == vector.end());
}

BOOST_AUTO_TEST_CASE(find_int2)
{
    int data[] = { 1, 2, 4, 5, 7, 8 };
    bc::vector<bc::int2_> vector(reinterpret_cast<bc::int2_ *>(data),
                                 reinterpret_cast<bc::int2_ *>(data) + 3);
    BOOST_CHECK_EQUAL(vector[0], bc::int2_(1, 2));
    BOOST_CHECK_EQUAL(vector[1], bc::int2_(4, 5));
    BOOST_CHECK_EQUAL(vector[2], bc::int2_(7, 8));

    bc::vector<bc::int2_>::iterator iter =
        bc::find(vector.begin(), vector.end(), bc::int2_(4, 5));
    BOOST_CHECK(iter == vector.begin() + 1);
    BOOST_CHECK_EQUAL(*iter, bc::int2_(4, 5));

    iter = bc::find(vector.begin(), vector.end(), bc::int2_(10, 11));
    BOOST_CHECK(iter == vector.end());
}

BOOST_AUTO_TEST_CASE(find_if_not_int)
{
    int data[] = { 2, 4, 6, 8, 1, 3, 5, 7, 9 };
    bc::vector<int> vector(data, data + 9);

    bc::vector<int>::iterator iter =
        bc::find_if_not(vector.begin(), vector.end(), bc::_1 == 2);
    BOOST_CHECK(iter == vector.begin() + 1);
    BOOST_CHECK_EQUAL(*iter, 4);
}

BOOST_AUTO_TEST_SUITE_END()
