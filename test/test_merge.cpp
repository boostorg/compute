//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestMerge
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/merge.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(simple_merge_int)
{
    int data1[] = { 1, 3, 5, 7 };
    int data2[] = { 2, 4, 6, 8 };

    boost::compute::vector<int> v1(data1, data1 + 4, context);
    boost::compute::vector<int> v2(data2, data2 + 4, context);
    boost::compute::vector<int> v3(8, context);

    // merge v1 with v2 into v3
    boost::compute::merge(
        v1.begin(), v1.end(),
        v2.begin(), v2.end(),
        v3.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 8, v3, (1, 2, 3, 4, 5, 6, 7, 8));

    // merge v2 with v1 into v3
    boost::compute::merge(
        v2.begin(), v2.end(),
        v1.begin(), v1.end(),
        v3.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 8, v3, (1, 2, 3, 4, 5, 6, 7, 8));

    // merge v1 with v1 into v3
    boost::compute::merge(
        v1.begin(), v1.end(),
        v1.begin(), v1.end(),
        v3.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 8, v3, (1, 1, 3, 3, 5, 5, 7, 7));

    // merge v2 with v2 into v3
    boost::compute::merge(
        v2.begin(), v2.end(),
        v2.begin(), v2.end(),
        v3.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 8, v3, (2, 2, 4, 4, 6, 6, 8, 8));

    // merge v1 with empty range into v3
    boost::compute::merge(
        v1.begin(), v1.end(),
        v1.begin(), v1.begin(),
        v3.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, v3, (1, 3, 5, 7));

    // merge v2 with empty range into v3
    boost::compute::merge(
        v1.begin(), v1.begin(),
        v2.begin(), v2.end(),
        v3.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, v3, (2, 4, 6, 8));
}

BOOST_AUTO_TEST_SUITE_END()
