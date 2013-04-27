//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestPartition
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/partition.hpp>
#include <boost/compute/algorithm/partition_copy.hpp>
#include <boost/compute/algorithm/is_partitioned.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(partition_float_vector)
{
    bc::vector<float> vector(context);
    vector.push_back(1.0f);
    vector.push_back(2.0f);
    vector.push_back(-1.0f);
    vector.push_back(-2.0f);
    vector.push_back(3.0f);
    vector.push_back(4.0f);
    vector.push_back(-3.0f);
    vector.push_back(-4.0f);

    // verify is_partitioned()
    BOOST_VERIFY(bc::is_partitioned(vector.begin(),
                                    vector.end(),
                                    bc::signbit_<float>(),
                                    queue) == false);

    // partition by signbit
    bc::vector<float>::iterator iter = bc::partition(vector.begin(),
                                                     vector.end(),
                                                     bc::signbit_<float>(),
                                                     queue);
    queue.finish();
    BOOST_VERIFY(iter == vector.begin() + 4);
    BOOST_CHECK_LT(vector[0], 0.0f);
    BOOST_CHECK_LT(vector[1], 0.0f);
    BOOST_CHECK_LT(vector[2], 0.0f);
    BOOST_CHECK_LT(vector[3], 0.0f);
    BOOST_CHECK_GT(vector[4], 0.0f);
    BOOST_CHECK_GT(vector[5], 0.0f);
    BOOST_CHECK_GT(vector[6], 0.0f);
    BOOST_CHECK_GT(vector[7], 0.0f);

    // verify is_partitioned()
    BOOST_VERIFY(bc::is_partitioned(vector.begin(),
                                    vector.end(),
                                    bc::signbit_<float>(),
                                    queue) == true);
}

BOOST_AUTO_TEST_CASE(partition_small_vector)
{
    bc::vector<float> vector(context);
    bc::partition(vector.begin(), vector.end(), bc::signbit_<float>(), queue);

    vector.push_back(1.0f);
    bc::partition(vector.begin(), vector.end(), bc::signbit_<float>(), queue);
    queue.finish();
    BOOST_CHECK_EQUAL(vector[0], 1.0f);

    vector.push_back(-1.0f);
    bc::partition(vector.begin(), vector.end(), bc::signbit_<float>(), queue);
    queue.finish();
    BOOST_CHECK_EQUAL(vector[0], -1.0f);
    BOOST_CHECK_EQUAL(vector[1], 1.0f);
}

BOOST_AUTO_TEST_SUITE_END()
