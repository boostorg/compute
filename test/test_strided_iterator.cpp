//---------------------------------------------------------------------------//
// Copyright (c) 2015 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestSkipIterator
#include <boost/test/unit_test.hpp>

#include <iterator>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/strided_iterator.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(value_type)
{
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::strided_iterator<
                boost::compute::buffer_iterator<int>
            >::value_type,
            int
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::strided_iterator<
                boost::compute::buffer_iterator<float>
            >::value_type,
            float
        >::value
    ));
}

BOOST_AUTO_TEST_CASE(base_type)
{
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::strided_iterator<
                boost::compute::buffer_iterator<int>
            >::base_type,
            boost::compute::buffer_iterator<int>
        >::value
    ));
}

BOOST_AUTO_TEST_CASE(copy)
{
    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    boost::compute::vector<int> vec(data, data + 8, queue);

    boost::compute::vector<int> result(4, context);

    // copy every other element to result
    boost::compute::copy(
        boost::compute::make_strided_iterator(vec.begin(), 2),
        boost::compute::make_strided_iterator(vec.end(), 2),
        result.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 4, result, (1, 3, 5, 7));

    // copy every other element to result
    boost::compute::copy(
        boost::compute::make_strided_iterator(vec.begin(), 3),
        boost::compute::make_strided_iterator(vec.begin()+6, 3),
        result.begin(),
        queue
    );
    CHECK_RANGE_EQUAL(int, 2, result, (1, 4));
}

BOOST_AUTO_TEST_CASE(distance)
{
    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    boost::compute::vector<int> vec(data, data + 8, queue);

    BOOST_CHECK_EQUAL(
         std::distance(
             boost::compute::make_strided_iterator(vec.begin(), 1),
             boost::compute::make_strided_iterator(vec.end(), 1)
         ),
         std::ptrdiff_t(8)
    );
    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_strided_iterator(vec.begin(), 2),
            boost::compute::make_strided_iterator(vec.end(), 2)
        ),
        std::ptrdiff_t(4)
    );
    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_strided_iterator(vec.begin(), 4),
            boost::compute::make_strided_iterator(vec.end(), 4)
        ),
        std::ptrdiff_t(2)
    );

    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_strided_iterator(vec.begin(), 3),
            boost::compute::make_strided_iterator(vec.begin()+6, 3)
        ),
        std::ptrdiff_t(2)
    );
}


BOOST_AUTO_TEST_SUITE_END()
