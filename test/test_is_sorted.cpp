//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestIsSorted
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/is_sorted.hpp>
#include <boost/compute/container/vector.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(is_sorted_int)
{
    bc::vector<int> vector;
    BOOST_VERIFY(bc::is_sorted(vector.begin(), vector.end()) == true);

    vector.push_back(1);
    BOOST_VERIFY(bc::is_sorted(vector.begin(), vector.end()) == true);

    vector.push_back(2);
    BOOST_VERIFY(bc::is_sorted(vector.begin(), vector.end()) == true);

    vector.push_back(0);
    BOOST_VERIFY(bc::is_sorted(vector.begin(), vector.end()) == false);

    vector.push_back(-2);
    BOOST_VERIFY(bc::is_sorted(vector.begin(), vector.end()) == false);

    bc::sort(vector.begin(), vector.end());
    BOOST_VERIFY(bc::is_sorted(vector.begin(), vector.end()) == true);
}

BOOST_AUTO_TEST_CASE(is_sorted_ones)
{
    bc::vector<int> vector(10);
    bc::fill(vector.begin(), vector.end(), int(1));
    BOOST_VERIFY(bc::is_sorted(vector.begin(), vector.end()) == true);
}
