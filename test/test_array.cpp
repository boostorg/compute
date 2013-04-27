//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestArray
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/array.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(concept_check)
{
    BOOST_CONCEPT_ASSERT((boost::Container<boost::compute::array<int, 3> >));
//    BOOST_CONCEPT_ASSERT((boost::SequenceConcept<boost::compute::array<int, 3> >));
    BOOST_CONCEPT_ASSERT((boost::RandomAccessIterator<boost::compute::array<int, 3>::iterator>));
    BOOST_CONCEPT_ASSERT((boost::RandomAccessIterator<boost::compute::array<int, 3>::const_iterator>));
}

BOOST_AUTO_TEST_CASE(size)
{
    boost::compute::array<int, 0> empty_array;
    BOOST_CHECK_EQUAL(empty_array.size(), size_t(0));

    boost::compute::array<int, 10> array10;
    BOOST_CHECK_EQUAL(array10.size(), size_t(10));
}

BOOST_AUTO_TEST_CASE(at)
{
    boost::compute::array<int, 3> array;
    array[0] = 3;
    array[1] = -2;
    array[2] = 5;
    BOOST_CHECK_EQUAL(array.at(0), 3);
    BOOST_CHECK_EQUAL(array.at(1), -2);
    BOOST_CHECK_EQUAL(array.at(2), 5);
    BOOST_CHECK_THROW(array.at(3), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(copy_from_vector)
{
    int data[] = { 3, 6, 9, 12 };
    boost::compute::vector<int> vector(data, data + 4, context);

    boost::compute::array<int, 4> array(context);
    boost::compute::copy(vector.begin(), vector.end(), array.begin());
    BOOST_CHECK_EQUAL(array[0], 3);
    BOOST_CHECK_EQUAL(array[1], 6);
    BOOST_CHECK_EQUAL(array[2], 9);
    BOOST_CHECK_EQUAL(array[3], 12);
}

BOOST_AUTO_TEST_CASE(fill)
{
    boost::compute::array<int, 4> array(context);
    array.fill(0);
    BOOST_CHECK_EQUAL(array[0], 0);
    BOOST_CHECK_EQUAL(array[1], 0);
    BOOST_CHECK_EQUAL(array[2], 0);
    BOOST_CHECK_EQUAL(array[3], 0);

    array.fill(17);
    BOOST_CHECK_EQUAL(array[0], 17);
    BOOST_CHECK_EQUAL(array[1], 17);
    BOOST_CHECK_EQUAL(array[2], 17);
    BOOST_CHECK_EQUAL(array[3], 17);
}

BOOST_AUTO_TEST_CASE(swap)
{
    int data[] = { 1, 2, 6, 9 };
    boost::compute::array<int, 4> a(context);
    boost::compute::copy(data, data + 4, a.begin());
    BOOST_CHECK_EQUAL(a[0], 1);
    BOOST_CHECK_EQUAL(a[1], 2);
    BOOST_CHECK_EQUAL(a[2], 6);
    BOOST_CHECK_EQUAL(a[3], 9);

    boost::compute::array<int, 4> b(context);
    b.fill(3);
    BOOST_CHECK_EQUAL(b[0], 3);
    BOOST_CHECK_EQUAL(b[1], 3);
    BOOST_CHECK_EQUAL(b[2], 3);
    BOOST_CHECK_EQUAL(b[3], 3);

    a.swap(b);
    BOOST_CHECK_EQUAL(a[0], 3);
    BOOST_CHECK_EQUAL(a[1], 3);
    BOOST_CHECK_EQUAL(a[2], 3);
    BOOST_CHECK_EQUAL(a[3], 3);
    BOOST_CHECK_EQUAL(b[0], 1);
    BOOST_CHECK_EQUAL(b[1], 2);
    BOOST_CHECK_EQUAL(b[2], 6);
    BOOST_CHECK_EQUAL(b[3], 9);
}

BOOST_AUTO_TEST_SUITE_END()
