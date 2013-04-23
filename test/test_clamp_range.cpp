//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestClampRange
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/detail/clamp_range.hpp>
#include <boost/compute/container/vector.hpp>

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(clamp_int_range)
{
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    compute::vector<int> input(data, data + 8, context);

    compute::vector<int> result(8, context);
    compute::detail::clamp_range(
        input.begin(),
        input.end(),
        result.begin(),
        3, // low
        6, // high
        queue
    );

    compute::copy(result.begin(), result.end(), data, queue);

    BOOST_CHECK_EQUAL(data[0], 3);
    BOOST_CHECK_EQUAL(data[1], 3);
    BOOST_CHECK_EQUAL(data[2], 3);
    BOOST_CHECK_EQUAL(data[3], 4);
    BOOST_CHECK_EQUAL(data[4], 5);
    BOOST_CHECK_EQUAL(data[5], 6);
    BOOST_CHECK_EQUAL(data[6], 6);
    BOOST_CHECK_EQUAL(data[7], 6);
}
