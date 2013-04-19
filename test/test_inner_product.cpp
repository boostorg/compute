//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestInnerProduct
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/inner_product.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(inner_product_int)
{
    int data1[] = { 1, 2, 3, 4 };
    bc::vector<int> input1(data1, data1 + 4, context);

    int data2[] = { 10, 20, 30, 40 };
    bc::vector<int> input2(data2, data2 + 4, context);

    int product = bc::inner_product(input1.begin(),
                                    input1.end(),
                                    input2.begin(),
                                    0,
                                    queue);
    BOOST_CHECK_EQUAL(product, 300);
}

BOOST_AUTO_TEST_SUITE_END()
