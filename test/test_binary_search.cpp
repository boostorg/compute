//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestBinarySearch
#include <boost/test/unit_test.hpp>

#include <iterator>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/binary_search.hpp>
#include <boost/compute/algorithm/lower_bound.hpp>
#include <boost/compute/algorithm/upper_bound.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(binary_search_int)
{
    int data[] = { 1, 2, 2, 2, 4, 4, 5, 7 };
    boost::compute::vector<int> vector(data, data + 8, queue);

    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(0), queue) == false);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(1), queue) == true);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(2), queue) == true);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(3), queue) == false);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(4), queue) == true);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(5), queue) == true);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(6), queue) == false);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(7), queue) == true);
    BOOST_CHECK(boost::compute::binary_search(vector.begin(), vector.end(), int(8), queue) == false);
}

BOOST_AUTO_TEST_CASE(range_bounds_int)
{
    int data[] = { 1, 2, 2, 2, 3, 3, 4, 5 };
    boost::compute::vector<int> vector(data, data + 8, queue);

    BOOST_CHECK(boost::compute::lower_bound(vector.begin(), vector.end(), int(0), queue) == vector.begin());
    BOOST_CHECK(boost::compute::upper_bound(vector.begin(), vector.end(), int(0), queue) == vector.begin());

    BOOST_CHECK(boost::compute::lower_bound(vector.begin(), vector.end(), int(1), queue) == vector.begin());
    BOOST_CHECK(boost::compute::upper_bound(vector.begin(), vector.end(), int(1), queue) == vector.begin() + 1);

    BOOST_CHECK(boost::compute::lower_bound(vector.begin(), vector.end(), int(2), queue) == vector.begin() + 1);
    BOOST_CHECK(boost::compute::upper_bound(vector.begin(), vector.end(), int(2), queue) == vector.begin() + 4);

    BOOST_CHECK(boost::compute::lower_bound(vector.begin(), vector.end(), int(3), queue) == vector.begin() + 4);
    BOOST_CHECK(boost::compute::upper_bound(vector.begin(), vector.end(), int(3), queue) == vector.begin() + 6);

    BOOST_CHECK(boost::compute::lower_bound(vector.begin(), vector.end(), int(4), queue) == vector.begin() + 6);
    BOOST_CHECK(boost::compute::upper_bound(vector.begin(), vector.end(), int(4), queue) == vector.begin() + 7);

    BOOST_CHECK(boost::compute::lower_bound(vector.begin(), vector.end(), int(5), queue) == vector.begin() + 7);
    BOOST_CHECK(boost::compute::upper_bound(vector.begin(), vector.end(), int(5), queue) == vector.end());

    BOOST_CHECK(boost::compute::lower_bound(vector.begin(), vector.end(), int(6), queue) == vector.end());
    BOOST_CHECK(boost::compute::upper_bound(vector.begin(), vector.end(), int(6), queue) == vector.end());
}

BOOST_AUTO_TEST_SUITE_END()
