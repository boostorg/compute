//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestString
#include <boost/test/unit_test.hpp>

#include <boost/compute/container/string.hpp>
#include <boost/compute/container/basic_string.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(empty)
{
    boost::compute::string str;
    BOOST_VERIFY(str.empty());
}

BOOST_AUTO_TEST_CASE(size)
{
    boost::compute::string str = "string";
    BOOST_VERIFY(!str.empty());
    BOOST_CHECK_EQUAL(str.size(), size_t(6));
    BOOST_CHECK_EQUAL(str.length(), size_t(6));
}

BOOST_AUTO_TEST_CASE(find)
{
    boost::compute::string str = "string";
    BOOST_VERIFY(!str.empty());
    BOOST_CHECK_EQUAL(str.find('r'), 2);
    BOOST_CHECK_NE(str.find('r'), 3);
}

BOOST_AUTO_TEST_SUITE_END()
