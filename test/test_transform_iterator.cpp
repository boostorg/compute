//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestTransformIterator
#include <boost/test/unit_test.hpp>

#include <iterator>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <boost/compute/types.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>

BOOST_AUTO_TEST_CASE(value_type)
{
    using boost::compute::float4_;

    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::transform_iterator<
                boost::compute::buffer_iterator<float>,
                boost::compute::sqrt<float>
            >::value_type,
            float
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::is_same<
            boost::compute::transform_iterator<
                boost::compute::buffer_iterator<float4_>,
                boost::compute::length<float4_>
            >::value_type,
            float
        >::value
    ));
}

BOOST_AUTO_TEST_CASE(copy)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    int data[] = { 1, -2, 3, -4, 5 };
    boost::compute::vector<int> a(data, data + 5, context);

    boost::compute::vector<int> b(5, context);
    boost::compute::copy(
        boost::compute::make_transform_iterator(
            a.begin(),
            boost::compute::abs<int>()
        ),
        boost::compute::make_transform_iterator(
            a.end(),
            boost::compute::abs<int>()
        ),
        b.begin(),
        queue
    );
    queue.finish();

    BOOST_CHECK_EQUAL(int(b[0]), 1);
    BOOST_CHECK_EQUAL(int(b[1]), 2);
    BOOST_CHECK_EQUAL(int(b[2]), 3);
    BOOST_CHECK_EQUAL(int(b[3]), 4);
    BOOST_CHECK_EQUAL(int(b[4]), 5);
}
