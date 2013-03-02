//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestMersenneTwister
#include <boost/test/unit_test.hpp>

#include <boost/compute/random/mersenne_twister.hpp>
#include <boost/compute/container/vector.hpp>

BOOST_AUTO_TEST_CASE(fill_uint)
{
    boost::compute::device gpu = boost::compute::system::default_device();
    boost::compute::context context(gpu);
    boost::compute::command_queue queue(context, gpu);

    using boost::compute::uint_;

    boost::compute::mt19937 rng(context);

    boost::compute::vector<uint_> vector(10, context);

    rng.fill(vector.begin(), vector.end(), queue);

    BOOST_CHECK_EQUAL(uint_(vector[0]), uint_(3499211612));
    BOOST_CHECK_EQUAL(uint_(vector[1]), uint_(581869302));
    BOOST_CHECK_EQUAL(uint_(vector[2]), uint_(3890346734));
    BOOST_CHECK_EQUAL(uint_(vector[3]), uint_(3586334585));
    BOOST_CHECK_EQUAL(uint_(vector[4]), uint_(545404204));
    BOOST_CHECK_EQUAL(uint_(vector[5]), uint_(4161255391));
    BOOST_CHECK_EQUAL(uint_(vector[6]), uint_(3922919429));
    BOOST_CHECK_EQUAL(uint_(vector[7]), uint_(949333985));
    BOOST_CHECK_EQUAL(uint_(vector[8]), uint_(2715962298));
    BOOST_CHECK_EQUAL(uint_(vector[9]), uint_(1323567403));
}
