//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestValarray
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/valarray.hpp>

BOOST_AUTO_TEST_CASE(size)
{
    boost::compute::valarray<float> array;
    BOOST_CHECK_EQUAL(array.size(), size_t(0));

    array.resize(10);
    BOOST_CHECK_EQUAL(array.size(), size_t(10));
}

BOOST_AUTO_TEST_CASE(at)
{
    int data[] = { 1, 2, 3, 4, 5 };
    boost::compute::valarray<int> array(data, 5);
    BOOST_CHECK_EQUAL(array.size(), size_t(5));

    BOOST_CHECK_EQUAL(int(array[0]), int(1));
    BOOST_CHECK_EQUAL(int(array[1]), int(2));
    BOOST_CHECK_EQUAL(int(array[2]), int(3));
    BOOST_CHECK_EQUAL(int(array[3]), int(4));
    BOOST_CHECK_EQUAL(int(array[4]), int(5));
}

BOOST_AUTO_TEST_CASE(min_and_max)
{
    int data[] = { 5, 2, 3, 7, 1, 9, 6, 5 };
    boost::compute::valarray<int> array(data, 8);
    BOOST_CHECK_EQUAL(array.size(), size_t(8));

    BOOST_CHECK_EQUAL((array.min)(), int(1));
    BOOST_CHECK_EQUAL((array.max)(), int(9));
}

BOOST_AUTO_TEST_CASE(sum)
{
    int data[] = { 1, 2, 3, 4 };
    boost::compute::valarray<int> array(data, 4);
    BOOST_CHECK_EQUAL(array.size(), size_t(4));

    BOOST_CHECK_EQUAL(array.sum(), int(10));
}
