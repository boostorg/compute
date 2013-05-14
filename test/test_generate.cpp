//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestGenerate
#include <boost/test/unit_test.hpp>

#include <boost/compute/function.hpp>
#include <boost/compute/algorithm/generate.hpp>
#include <boost/compute/algorithm/generate_n.hpp>
#include <boost/compute/container/vector.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(generate4)
{
    bc::vector<int> vector(4);
    bc::fill(vector.begin(), vector.end(), 2);
    CHECK_RANGE_EQUAL(int, 4, vector, (2, 2, 2, 2));

    bc::function<int (void)> ret4 =
        bc::make_function_from_source<int ()>("ret4", "int ret4() { return 4; }");
    bc::generate(vector.begin(), vector.end(), ret4);
    CHECK_RANGE_EQUAL(int, 4, vector, (4, 4, 4, 4));
}

BOOST_AUTO_TEST_SUITE_END()
