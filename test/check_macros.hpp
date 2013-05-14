//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_TEST_CHECK_MACROS_HPP
#define BOOST_COMPUTE_TEST_CHECK_MACROS_HPP

#define LIST_ARRAY_VALUES(z, n, data) \
    BOOST_PP_COMMA_IF(n) BOOST_PP_ARRAY_ELEM(n, data)

// checks 'size' values of 'type' in the device range '_actual`
// against the values given in the array '_expected'
#define CHECK_RANGE_EQUAL(type, size, _actual, _expected) \
    { \
        type actual[size]; \
        boost::compute::copy( \
            _actual.begin(), _actual.end(), actual, queue \
        ); \
        const type expected[size] = { \
            BOOST_PP_REPEAT(size, LIST_ARRAY_VALUES, (size, _expected)) \
        }; \
        BOOST_CHECK_EQUAL_COLLECTIONS( \
            actual, actual + size, expected, expected + size \
        ); \
    }

#endif // BOOST_COMPUTE_TEST_CHECK_MACROS_HPP
