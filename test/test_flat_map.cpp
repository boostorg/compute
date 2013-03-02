//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestFlatMap
#include <boost/test/unit_test.hpp>

#include <utility>

#include <boost/concept_check.hpp>

#include <boost/compute/container/flat_map.hpp>

BOOST_AUTO_TEST_CASE(concept_check)
{
    BOOST_CONCEPT_ASSERT((boost::Container<boost::compute::flat_map<int, float> >));
//    BOOST_CONCEPT_ASSERT((boost::SimpleAssociativeContainer<boost::compute::flat_map<int, float> >));
//    BOOST_CONCEPT_ASSERT((boost::UniqueAssociativeContainer<boost::compute::flat_map<int, float> >));
    BOOST_CONCEPT_ASSERT((boost::RandomAccessIterator<boost::compute::flat_map<int, float>::iterator>));
    BOOST_CONCEPT_ASSERT((boost::RandomAccessIterator<boost::compute::flat_map<int, float>::const_iterator>));
}

BOOST_AUTO_TEST_CASE(insert)
{
    boost::compute::flat_map<int, float> map;
    map.insert(std::make_pair(1, 1.1f));
    map.insert(std::make_pair(-1, -1.1f));
    map.insert(std::make_pair(3, 3.3f));
    map.insert(std::make_pair(2, 2.2f));
    BOOST_CHECK_EQUAL(map.size(), size_t(4));
    BOOST_CHECK(map.find(-1) == map.begin() + 0);
    BOOST_CHECK(map.find(1) == map.begin() + 1);
    BOOST_CHECK(map.find(2) == map.begin() + 2);
    BOOST_CHECK(map.find(3) == map.begin() + 3);

    map.insert(std::make_pair(2, -2.2f));
    BOOST_CHECK_EQUAL(map.size(), size_t(4));
}

BOOST_AUTO_TEST_CASE(at)
{
    boost::compute::flat_map<int, float> map;
    map.insert(std::make_pair(1, 1.1f));
    map.insert(std::make_pair(4, 4.4f));
    map.insert(std::make_pair(3, 3.3f));
    map.insert(std::make_pair(2, 2.2f));
    BOOST_CHECK_EQUAL(float(map.at(1)), float(1.1f));
    BOOST_CHECK_EQUAL(float(map.at(2)), float(2.2f));
    BOOST_CHECK_EQUAL(float(map.at(3)), float(3.3f));
    BOOST_CHECK_EQUAL(float(map.at(4)), float(4.4f));
}

BOOST_AUTO_TEST_CASE(index_operator)
{
    boost::compute::flat_map<int, float> map;
    map[1] = 1.1f;
    map[2] = 2.2f;
    map[3] = 3.3f;
    map[4] = 4.4f;
    BOOST_CHECK_EQUAL(float(map[1]), float(1.1f));
    BOOST_CHECK_EQUAL(float(map[2]), float(2.2f));
    BOOST_CHECK_EQUAL(float(map[3]), float(3.3f));
    BOOST_CHECK_EQUAL(float(map[4]), float(4.4f));
}

BOOST_AUTO_TEST_CASE(simple_histogram)
{
    int data[] = { 1, 5, 2, 9, 3, 2, 6, 4, 7, 4,
                   9, 0, 2, 1, 3, 5, 6, 2, 9, 1,
                   1, 4, 2, 3, 4, 6, 7, 8, 9, 2 };

    boost::compute::flat_map<int, int> map;

    for(size_t i = 0; i < 30; i++){
        map[data[i]]++;
    }

    BOOST_CHECK_EQUAL(int(map[0]), int(1));
    BOOST_CHECK_EQUAL(int(map[1]), int(4));
    BOOST_CHECK_EQUAL(int(map[2]), int(6));
    BOOST_CHECK_EQUAL(int(map[3]), int(3));
    BOOST_CHECK_EQUAL(int(map[4]), int(4));
    BOOST_CHECK_EQUAL(int(map[5]), int(2));
    BOOST_CHECK_EQUAL(int(map[6]), int(3));
    BOOST_CHECK_EQUAL(int(map[7]), int(2));
    BOOST_CHECK_EQUAL(int(map[8]), int(1));
    BOOST_CHECK_EQUAL(int(map[9]), int(4));
}
