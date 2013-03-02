//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestScatter
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/scatter.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/constant_buffer_iterator.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(scatter_int)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);

    int input_data[] = { 1, 2, 3, 4, 5 };
    bc::vector<int> input(input_data, input_data + 5, context);

    int map_data[] = { 0, 4, 1, 3, 2 };
    bc::vector<int> map(map_data, map_data + 5, context);

    bc::vector<int> output(5, context);
    bc::scatter(input.begin(), input.end(), map.begin(), output.begin());
    BOOST_CHECK_EQUAL(output[0], 1);
    BOOST_CHECK_EQUAL(output[1], 3);
    BOOST_CHECK_EQUAL(output[2], 5);
    BOOST_CHECK_EQUAL(output[3], 4);
    BOOST_CHECK_EQUAL(output[4], 2);
}

BOOST_AUTO_TEST_CASE(scatter_constant_indices)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);

    int input_data[] = { 1, 2, 3, 4, 5 };
    bc::vector<int> input(input_data, input_data + 5, context);

    int map_data[] = { 0, 4, 1, 3, 2 };
    bc::buffer map_buffer(context,
                          5 * sizeof(int),
                          bc::buffer::read_only | bc::buffer::use_host_ptr,
                          map_data);

    bc::vector<int> output(5, context);
    bc::scatter(input.begin(),
                input.end(),
                bc::make_constant_buffer_iterator<int>(map_buffer, 0),
                output.begin());
    BOOST_CHECK_EQUAL(output[0], 1);
    BOOST_CHECK_EQUAL(output[1], 3);
    BOOST_CHECK_EQUAL(output[2], 5);
    BOOST_CHECK_EQUAL(output[3], 4);
    BOOST_CHECK_EQUAL(output[4], 2);
}
