//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestStableSort
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/stable_sort.hpp>
#include <boost/compute/algorithm/is_sorted.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(sort_int_vector)
{
    int data[] = { -4, 152, -5000, 963, 75321, -456, 0, 1112 };
    boost::compute::vector<int> vector(data, data + 8);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end()) == false);

    boost::compute::stable_sort(vector.begin(), vector.end());
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end()) == true);

    boost::compute::copy(vector.begin(), vector.end(), data);
    BOOST_CHECK_EQUAL(data[0], -5000);
    BOOST_CHECK_EQUAL(data[1], -456);
    BOOST_CHECK_EQUAL(data[2], -4);
    BOOST_CHECK_EQUAL(data[3], 0);
    BOOST_CHECK_EQUAL(data[4], 152);
    BOOST_CHECK_EQUAL(data[5], 963);
    BOOST_CHECK_EQUAL(data[6], 1112);
    BOOST_CHECK_EQUAL(data[7], 75321);
}

BOOST_AUTO_TEST_SUITE_END()
