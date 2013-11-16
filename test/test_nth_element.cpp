//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestNthElement
#include <boost/test/unit_test.hpp>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy_n.hpp>
#include <boost/compute/algorithm/nth_element.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(nth_element_int)
{
    int data[] = { 9, 15, 1, 4, 9, 9, 4, 15, 12, 1 };
    boost::compute::vector<int> vector(10, context);

    boost::compute::copy_n(data, 10, vector.begin(), queue);

    boost::compute::nth_element(
        vector.begin(), vector.begin() + 5, vector.end(), queue
    );
    CHECK_RANGE_EQUAL(int, 5, vector, (1, 1, 4, 4, 9));

    boost::compute::nth_element(
        vector.begin(), vector.end(), vector.end(), queue
    );
    CHECK_RANGE_EQUAL(int, 10, vector, (1, 1, 4, 4, 9, 9, 9, 12, 15, 15));
}

BOOST_AUTO_TEST_CASE(nth_element_median)
{
    int data[] = { 5, 6, 4, 3, 2, 6, 7, 9, 3 };
    boost::compute::vector<int> v(9, context);
    boost::compute::copy_n(data, 9, v.begin(), queue);

    boost::compute::nth_element(v.begin(), v.begin() + v.size()/2, v.end(), queue);

    boost::compute::copy_n(v.begin(), 9, data, queue);
    BOOST_CHECK_EQUAL(data[v.size()/2], 5);
}

BOOST_AUTO_TEST_SUITE_END()
