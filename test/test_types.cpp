//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestTypes
#include <boost/test/unit_test.hpp>

#include <string>
#include <sstream>

#include <boost/compute/types/builtin.hpp>

BOOST_AUTO_TEST_CASE(vector_ctor)
{
    boost::compute::int4_ i4(1, 2, 3, 4);
    BOOST_CHECK(i4 == boost::compute::int4_(1, 2, 3, 4));
    BOOST_CHECK_EQUAL(i4, boost::compute::int4_(1, 2, 3, 4));
    BOOST_CHECK_EQUAL(i4[0], 1);
    BOOST_CHECK_EQUAL(i4[1], 2);
    BOOST_CHECK_EQUAL(i4[2], 3);
    BOOST_CHECK_EQUAL(i4[3], 4);
}

BOOST_AUTO_TEST_CASE(vector_accessors)
{
    boost::compute::int4_ i4;
    i4.s0 = 1; i4.s1 = 2; i4.s2 = 3; i4.s3 = 4;
    BOOST_CHECK_EQUAL(i4, boost::compute::int4_(1, 2, 3, 4));
    BOOST_CHECK_EQUAL(i4.s0, 1);
    BOOST_CHECK_EQUAL(i4.s1, 2);
    BOOST_CHECK_EQUAL(i4.s2, 3);
    BOOST_CHECK_EQUAL(i4.s3, 4);

    i4 = boost::compute::int4_(4,3,2,1);
    BOOST_CHECK_EQUAL(i4.x, 4);
    BOOST_CHECK_EQUAL(i4.y, 3);
    BOOST_CHECK_EQUAL(i4.z, 2);
    BOOST_CHECK_EQUAL(i4.w, 1);    
}

BOOST_AUTO_TEST_CASE(vector_string)
{
    std::stringstream stream;
    stream << boost::compute::int2_(1, 2);
    BOOST_CHECK_EQUAL(stream.str(), std::string("int2(1, 2)"));
}
