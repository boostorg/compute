//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestGather
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/gather.hpp>
#include <boost/compute/container/vector.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(gather_int)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    int input_data[] = { 1, 2, 3, 4, 5 };
    bc::vector<int> input(input_data, input_data + 5, context);

    int map_data[] = { 0, 4, 1, 3, 2 };
    bc::vector<int> map(map_data, map_data + 5, context);

    bc::vector<int> output(5, context);
    bc::gather(input.begin(), input.end(), map.begin(), output.begin());
    BOOST_CHECK_EQUAL(output[0], 1);
    BOOST_CHECK_EQUAL(output[1], 5);
    BOOST_CHECK_EQUAL(output[2], 2);
    BOOST_CHECK_EQUAL(output[3], 4);
    BOOST_CHECK_EQUAL(output[4], 3);
}
