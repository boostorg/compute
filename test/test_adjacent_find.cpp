//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAdjacentFind
#include <boost/test/unit_test.hpp>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/adjacent_find.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(adjacent_find_int)
{
    int data[] = { 1, 3, 5, 5, 6, 7, 7, 8 };
    boost::compute::vector<int> vector(data, data + 8);

    boost::compute::vector<int>::iterator iter =
        boost::compute::adjacent_find(vector.begin(), vector.end());
    BOOST_VERIFY(iter == vector.begin() + 2);
}

BOOST_AUTO_TEST_SUITE_END()
