//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAnyAllNoneOf
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/algorithm/all_of.hpp>
#include <boost/compute/algorithm/any_of.hpp>
#include <boost/compute/algorithm/none_of.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(any_all_none_of)
{
    int data[] = { 1, 2, 3, 4, 5, 6 };
    bc::vector<int> v(data, data + 6);

    using ::boost::compute::_1;

    BOOST_CHECK(bc::any_of(v.begin(), v.end(), _1 == 6) == true);
    BOOST_CHECK(bc::any_of(v.begin(), v.end(), _1 == 9) == false);
    BOOST_CHECK(bc::none_of(v.begin(), v.end(), _1 == 6) == false);
    BOOST_CHECK(bc::none_of(v.begin(), v.end(), _1 == 9) == true);
    BOOST_CHECK(bc::all_of(v.begin(), v.end(), _1 == 6) == false);
    BOOST_CHECK(bc::all_of(v.begin(), v.end(), _1 < 9) == true);
    BOOST_CHECK(bc::all_of(v.begin(), v.end(), _1 < 6) == false);
    BOOST_CHECK(bc::all_of(v.begin(), v.end(), _1 >= 1) == true);
}

BOOST_AUTO_TEST_SUITE_END()
