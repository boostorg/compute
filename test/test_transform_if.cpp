//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestTransformIf
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/experimental/transform_if.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(abs_if_odd)
{
    using compute::lambda::_1;

    // input data
    int data[] = { -2, -3, -4, -5, -6, -7, -8, -9 };
    compute::vector<int> vector(data, data + 8, queue);

    // calculate absolute value only for odd values
    compute::experimental::transform_if(
        vector.begin(),
        vector.end(),
        vector.begin(),
        compute::abs<int>(),
        _1 % 2 != 0,
        queue
    );

    // check transformed values
    CHECK_RANGE_EQUAL(int, 8, vector, (-2, +3, -4, +5, -6, +7, -8, +9));
}

BOOST_AUTO_TEST_SUITE_END()
