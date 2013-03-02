//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestMismatch
#include <boost/test/unit_test.hpp>

#include <boost/compute/algorithm/mismatch.hpp>
#include <boost/compute/container/vector.hpp>

BOOST_AUTO_TEST_CASE(mismatch_int)
{
    int data1[] = { 1, 2, 3, 4, 5, 6 };
    int data2[] = { 1, 2, 3, 7, 5, 6 };

    boost::compute::vector<int> vector1(data1, data1 + 6);
    boost::compute::vector<int> vector2(data2, data2 + 6);

    typedef boost::compute::vector<int>::iterator iter;

    std::pair<iter, iter> location =
        boost::compute::mismatch(vector1.begin(), vector1.end(), vector2.begin());
    BOOST_CHECK(location.first == vector1.begin() + 3);
    BOOST_CHECK_EQUAL(int(*location.first), int(4));
    BOOST_CHECK(location.second == vector2.begin() + 3);
    BOOST_CHECK_EQUAL(int(*location.second), int(7));
}
