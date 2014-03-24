//---------------------------------------------------------------------------//
// Copyright (c) 2014 Roshan <thisisroshansmail@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestUnique
#include <boost/test/unit_test.hpp>

#include <boost/compute/algorithm/unique.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;
namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(unique_int)
{
    int data[] = {1, 6, 6, 4, 2, 2, 4};

    bc::vector<int> input(data, data + 7);
    
    bc::vector<int>::iterator iter =
        bc::unique(input.begin(), input.end());
    
    BOOST_VERIFY(iter == input.begin() + 5);
    CHECK_RANGE_EQUAL(int, 7, input, (1, 6, 4, 2, 4, 2, 4));
}

BOOST_AUTO_TEST_SUITE_END()
