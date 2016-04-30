//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jason Rhinelander <jason@imaginary.ca>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestLiteralConversion
#include <boost/test/unit_test.hpp>

#include <string>
#include <sstream>
#include <vector>

#include <boost/compute/detail/literal.hpp>

BOOST_AUTO_TEST_CASE(literal_conversion_float)
{
    std::vector<float> values, roundtrip;
    values.push_back(1.2345679f);
    values.push_back(1.2345680f);
    values.push_back(1.2345681f);
    for (size_t i = 0; i < values.size(); i++) {
        std::istringstream iss(boost::compute::detail::make_literal(values[i]));
        float x;
        BOOST_CHECK(iss >> x);
        BOOST_CHECK_EQUAL(char(iss.get()), 'f');
        // Make sure we're at the end:
        iss.peek();
        BOOST_CHECK(iss.eof());

        roundtrip.push_back(x);
    }
    BOOST_CHECK_EQUAL(values[0], roundtrip[0]);
    BOOST_CHECK_EQUAL(values[1], roundtrip[1]);
    BOOST_CHECK_EQUAL(values[2], roundtrip[2]);
}

BOOST_AUTO_TEST_CASE(literal_conversion_double)
{
    std::vector<double> values, roundtrip;
    values.push_back(1.2345678901234567);
    values.push_back(1.2345678901234569);
    values.push_back(1.2345678901234571);
    for (size_t i = 0; i < values.size(); i++) {
        std::istringstream iss(boost::compute::detail::make_literal(values[i]));
        double x;
        BOOST_CHECK(iss >> x);
        // Make sure we're at the end:
        iss.peek();
        BOOST_CHECK(iss.eof());
        roundtrip.push_back(x);
    }
    BOOST_CHECK_EQUAL(values[0], roundtrip[0]);
    BOOST_CHECK_EQUAL(values[1], roundtrip[1]);
    BOOST_CHECK_EQUAL(values[2], roundtrip[2]);
}
