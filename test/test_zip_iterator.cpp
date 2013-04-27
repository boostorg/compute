//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestZipIterator
#include <boost/test/unit_test.hpp>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <boost/compute/tuple.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(value_type)
{
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::zip_iterator<
                boost::tuple<
                    boost::compute::buffer_iterator<float>,
                    boost::compute::buffer_iterator<int>
                >
            >::value_type,
            boost::tuple<float, int>
        >::value
    ));
}

BOOST_AUTO_TEST_CASE(distance)
{
    boost::compute::vector<char> char_vector(5, context);
    boost::compute::vector<int> int_vector(5, context);

    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_zip_iterator(
                boost::make_tuple(
                    char_vector.begin(),
                    int_vector.begin()
                )
            ),
            boost::compute::make_zip_iterator(
                boost::make_tuple(
                    char_vector.end(),
                    int_vector.end()
                )
            )
        ),
        ptrdiff_t(5)
    );

    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_zip_iterator(
                boost::make_tuple(
                    char_vector.begin(),
                    int_vector.begin()
                )
            ) + 1,
            boost::compute::make_zip_iterator(
                boost::make_tuple(
                    char_vector.end(),
                    int_vector.end()
                )
            ) - 1
        ),
        ptrdiff_t(3)
    );

    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_zip_iterator(
                boost::make_tuple(
                    char_vector.begin() + 2,
                    int_vector.begin() + 2
                )
            ),
            boost::compute::make_zip_iterator(
                boost::make_tuple(
                    char_vector.end() - 1,
                    int_vector.end() - 1
                )
            )
        ),
        ptrdiff_t(2)
    );
}

BOOST_AUTO_TEST_CASE(copy)
{
    // create three separate vectors of three different types
    char char_data[] = { 'x', 'y', 'z' };
    boost::compute::vector<char> char_vector(char_data, char_data + 3, context);

    int int_data[] = { 4, 7, 9 };
    boost::compute::vector<int> int_vector(int_data, int_data + 3, context);

    float float_data[] = { 3.2f, 4.5f, 7.6f };
    boost::compute::vector<float> float_vector(float_data, float_data + 3, context);

    // zip all three vectors into a single tuple vector
    boost::compute::vector<boost::tuple<char, int, float> > tuple_vector(3, context);

    boost::compute::copy(
        boost::compute::make_zip_iterator(
            boost::make_tuple(
                char_vector.begin(),
                int_vector.begin(),
                float_vector.begin()
            )
        ),
        boost::compute::make_zip_iterator(
            boost::make_tuple(
                char_vector.end(),
                int_vector.end(),
                float_vector.end()
            )
        ),
        tuple_vector.begin()
    );

    // copy tuple vector to host
    std::vector<boost::tuple<char, int, float> > host_vector(3);

    boost::compute::copy(
        tuple_vector.begin(),
        tuple_vector.end(),
        host_vector.begin()
    );

    // check tuple values
    BOOST_CHECK_EQUAL(host_vector[0], boost::make_tuple('x', 4, 3.2f));
    BOOST_CHECK_EQUAL(host_vector[1], boost::make_tuple('y', 7, 4.5f));
    BOOST_CHECK_EQUAL(host_vector[2], boost::make_tuple('z', 9, 7.6f));
}

BOOST_AUTO_TEST_SUITE_END()
