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

BOOST_AUTO_TEST_CASE(simple_merge_int)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

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

    BOOST_CHECK_EQUAL(int(v3[0]), 1);
    BOOST_CHECK_EQUAL(int(v3[1]), 2);
    BOOST_CHECK_EQUAL(int(v3[2]), 3);
    BOOST_CHECK_EQUAL(int(v3[3]), 4);
    BOOST_CHECK_EQUAL(int(v3[4]), 5);
    BOOST_CHECK_EQUAL(int(v3[5]), 6);
    BOOST_CHECK_EQUAL(int(v3[6]), 7);
    BOOST_CHECK_EQUAL(int(v3[7]), 8);

    // merge v2 with v1 into v3
    boost::compute::merge(
        v2.begin(), v2.end(),
        v1.begin(), v1.end(),
        v3.begin(),
        queue
    );

    BOOST_CHECK_EQUAL(int(v3[0]), 1);
    BOOST_CHECK_EQUAL(int(v3[1]), 2);
    BOOST_CHECK_EQUAL(int(v3[2]), 3);
    BOOST_CHECK_EQUAL(int(v3[3]), 4);
    BOOST_CHECK_EQUAL(int(v3[4]), 5);
    BOOST_CHECK_EQUAL(int(v3[5]), 6);
    BOOST_CHECK_EQUAL(int(v3[6]), 7);
    BOOST_CHECK_EQUAL(int(v3[7]), 8);

    // merge v1 with v1 into v3
    boost::compute::merge(
        v1.begin(), v1.end(),
        v1.begin(), v1.end(),
        v3.begin(),
        queue
    );

    BOOST_CHECK_EQUAL(int(v3[0]), 1);
    BOOST_CHECK_EQUAL(int(v3[1]), 1);
    BOOST_CHECK_EQUAL(int(v3[2]), 3);
    BOOST_CHECK_EQUAL(int(v3[3]), 3);
    BOOST_CHECK_EQUAL(int(v3[4]), 5);
    BOOST_CHECK_EQUAL(int(v3[5]), 5);
    BOOST_CHECK_EQUAL(int(v3[6]), 7);
    BOOST_CHECK_EQUAL(int(v3[7]), 7);

    // merge v2 with v2 into v3
    boost::compute::merge(
        v2.begin(), v2.end(),
        v2.begin(), v2.end(),
        v3.begin(),
        queue
    );

    BOOST_CHECK_EQUAL(int(v3[0]), 2);
    BOOST_CHECK_EQUAL(int(v3[1]), 2);
    BOOST_CHECK_EQUAL(int(v3[2]), 4);
    BOOST_CHECK_EQUAL(int(v3[3]), 4);
    BOOST_CHECK_EQUAL(int(v3[4]), 6);
    BOOST_CHECK_EQUAL(int(v3[5]), 6);
    BOOST_CHECK_EQUAL(int(v3[6]), 8);
    BOOST_CHECK_EQUAL(int(v3[7]), 8);

    // merge v1 with empty range into v3
    boost::compute::merge(
        v1.begin(), v1.end(),
        v1.begin(), v1.begin(),
        v3.begin(),
        queue
    );

    BOOST_CHECK_EQUAL(int(v3[0]), 1);
    BOOST_CHECK_EQUAL(int(v3[1]), 3);
    BOOST_CHECK_EQUAL(int(v3[2]), 5);
    BOOST_CHECK_EQUAL(int(v3[3]), 7);

    // merge v2 with empty range into v3
    boost::compute::merge(
        v1.begin(), v1.begin(),
        v2.begin(), v2.end(),
        v3.begin(),
        queue
    );

    BOOST_CHECK_EQUAL(int(v3[0]), 2);
    BOOST_CHECK_EQUAL(int(v3[1]), 4);
    BOOST_CHECK_EQUAL(int(v3[2]), 6);
    BOOST_CHECK_EQUAL(int(v3[3]), 8);
}
