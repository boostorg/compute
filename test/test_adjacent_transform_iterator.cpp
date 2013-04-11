//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAdjacentTransformIterator
#include <boost/test/unit_test.hpp>

#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/max_element.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/detail/adjacent_transform_iterator.hpp>

BOOST_AUTO_TEST_CASE(copy)
{
    int data[] = { 1, 2, 4, 7, 11, 16 };
    boost::compute::vector<int> input(data, data + 6);

    boost::compute::vector<int> output(6);

    boost::compute::minus<int> minus_op;

    boost::compute::copy(
        boost::compute::detail::make_adjacent_transform_iterator(input.begin(), minus_op),
        boost::compute::detail::make_adjacent_transform_iterator(input.end(), minus_op),
        output.begin()
    );
    BOOST_CHECK_EQUAL(int(output[0]), int(1));
    BOOST_CHECK_EQUAL(int(output[1]), int(1));
    BOOST_CHECK_EQUAL(int(output[2]), int(2));
    BOOST_CHECK_EQUAL(int(output[3]), int(3));
    BOOST_CHECK_EQUAL(int(output[4]), int(4));
    BOOST_CHECK_EQUAL(int(output[5]), int(5));
}

BOOST_AUTO_TEST_CASE(find_largest_gap)
{
    float data[] = { 2.0f, 4.0f, 8.0f, 10.0f, 12.0f };
    boost::compute::vector<float> vector(data, data + 5);

    boost::compute::minus<float> minus_op;

    boost::compute::vector<float>::iterator iter =
        boost::compute::max_element(
            boost::compute::detail::make_adjacent_transform_iterator(vector.begin(), minus_op),
            boost::compute::detail::make_adjacent_transform_iterator(vector.end(), minus_op)
        ).base() - 1;
    BOOST_VERIFY(iter == vector.begin() + 1);
}
