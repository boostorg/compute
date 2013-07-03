//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestFill
#include <boost/test/unit_test.hpp>

#include <boost/compute/future.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/algorithm/fill_n.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;
namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(fill_int)
{
    bc::vector<int> vector(1000);
    bc::fill(vector.begin(), vector.end(), 0);
    bc::system::finish();
    BOOST_CHECK_EQUAL(vector.front(), 0);
    BOOST_CHECK_EQUAL(vector.back(), 0);

    bc::fill(vector.begin(), vector.end(), 100);
    bc::system::finish();
    BOOST_CHECK_EQUAL(vector.front(), 100);
    BOOST_CHECK_EQUAL(vector.back(), 100);

    bc::fill(vector.begin() + 500, vector.end(), 42);
    bc::system::finish();
    BOOST_CHECK_EQUAL(vector.front(), 100);
    BOOST_CHECK_EQUAL(vector[499], 100);
    BOOST_CHECK_EQUAL(vector[500], 42);
    BOOST_CHECK_EQUAL(vector.back(), 42);
}

BOOST_AUTO_TEST_CASE(fill_int2)
{
    using bc::int2_;

    bc::vector<int2_> vector(10);
    bc::fill(vector.begin(), vector.end(), int2_(4, 2));
    CHECK_RANGE_EQUAL(int2_, 10, vector,
        (int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2),
         int2_(4, 2))
    );

    bc::fill(vector.begin(), vector.end(), int2_(-2, -4));
    CHECK_RANGE_EQUAL(int2_, 10, vector,
        (int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4),
         int2_(-2, -4))
    );
}

BOOST_AUTO_TEST_CASE(fill_n_float)
{
    bc::vector<float> vector(4);
    bc::fill_n(vector.begin(), 4, 1.5f);
    CHECK_RANGE_EQUAL(float, 4, vector, (1.5f, 1.5f, 1.5f, 1.5f));

    bc::fill_n(vector.begin(), 3, 2.75f);
    CHECK_RANGE_EQUAL(float, 4, vector, (2.75f, 2.75f, 2.75f, 1.5f));

    bc::fill_n(vector.begin() + 1, 2, -3.2f);
    CHECK_RANGE_EQUAL(float, 4, vector, (2.75f, -3.2f, -3.2f, 1.5f));

    bc::fill_n(vector.begin(), 4, 0.0f);
    CHECK_RANGE_EQUAL(float, 4, vector, (0.0f, 0.0f, 0.0f, 0.0f));
}

BOOST_AUTO_TEST_CASE(check_fill_type)
{
    compute::vector<int> vector(5, context);
    compute::future<void> future =
        compute::fill_async(vector.begin(), vector.end(), 42, queue);
    future.wait();

    #ifdef CL_VERSION_1_2
    BOOST_CHECK(
        future.get_event().get_command_type() == CL_COMMAND_FILL_BUFFER
    );
    #else
    BOOST_CHECK(
        future.get_event().get_command_type() == CL_COMMAND_NDRANGE_KERNEL
    );
    #endif
}

BOOST_AUTO_TEST_SUITE_END()
