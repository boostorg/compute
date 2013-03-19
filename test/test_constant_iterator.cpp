//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestConstantIterator
#include <boost/test/unit_test.hpp>

#include <iterator>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/constant_iterator.hpp>

BOOST_AUTO_TEST_CASE(value_type)
{
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::constant_iterator<int>::value_type,
            int
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::constant_iterator<float>::value_type,
            float
        >::value
    ));
}

BOOST_AUTO_TEST_CASE(distance)
{
    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_constant_iterator(128, 0),
            boost::compute::make_constant_iterator(128, 10)
        ),
        std::ptrdiff_t(10)
    );
    BOOST_CHECK_EQUAL(
        std::distance(
            boost::compute::make_constant_iterator(256, 5),
            boost::compute::make_constant_iterator(256, 10)
        ),
        std::ptrdiff_t(5)
    );
}

BOOST_AUTO_TEST_CASE(copy)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    boost::compute::vector<int> vector(10, context);

    boost::compute::copy(
        boost::compute::make_constant_iterator(42, 0),
        boost::compute::make_constant_iterator(42, 10),
        vector.begin(),
        queue
    );
    queue.finish();

    BOOST_CHECK_EQUAL(int(vector[0]), 42);
    BOOST_CHECK_EQUAL(int(vector[1]), 42);
    BOOST_CHECK_EQUAL(int(vector[2]), 42);
    BOOST_CHECK_EQUAL(int(vector[3]), 42);
    BOOST_CHECK_EQUAL(int(vector[4]), 42);
    BOOST_CHECK_EQUAL(int(vector[5]), 42);
    BOOST_CHECK_EQUAL(int(vector[6]), 42);
    BOOST_CHECK_EQUAL(int(vector[7]), 42);
    BOOST_CHECK_EQUAL(int(vector[8]), 42);
    BOOST_CHECK_EQUAL(int(vector[9]), 42);
}
