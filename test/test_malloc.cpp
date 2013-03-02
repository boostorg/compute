//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestMalloc
#include <boost/test/unit_test.hpp>

#include <boost/compute/malloc.hpp>
#include <boost/compute/algorithm/copy.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(malloc_int)
{
    bc::device_ptr<int> ptr = bc::malloc<int>(5);

    int input_data[] = { 2, 5, 8, 3, 6 };
    bc::copy(input_data, input_data + 5, ptr);

    int output_data[5];
    bc::copy(ptr, ptr + 5, output_data);

    BOOST_CHECK_EQUAL(output_data[0], 2);
    BOOST_CHECK_EQUAL(output_data[1], 5);
    BOOST_CHECK_EQUAL(output_data[2], 8);
    BOOST_CHECK_EQUAL(output_data[3], 3);
    BOOST_CHECK_EQUAL(output_data[4], 6);

    bc::free(ptr);
}
