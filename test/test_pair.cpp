//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestPair
#include <boost/test/unit_test.hpp>

#include <boost/compute/pair.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>

BOOST_AUTO_TEST_CASE(vector_pair_int_float)
{
    boost::compute::vector<std::pair<int, float> > vector;
    vector.push_back(std::make_pair(1, 1.1f));
    vector.push_back(std::make_pair(2, 2.2f));
    vector.push_back(std::make_pair(3, 3.3f));
    BOOST_CHECK_EQUAL(vector.size(), size_t(3));
    BOOST_CHECK(vector[0] == std::make_pair(1, 1.1f));
    BOOST_CHECK(vector[1] == std::make_pair(2, 2.2f));
    BOOST_CHECK(vector[2] == std::make_pair(3, 3.3f));
}

BOOST_AUTO_TEST_CASE(copy_pair_vector)
{
    boost::compute::vector<std::pair<int, float> > input;
    input.push_back(std::make_pair(1, 2.0f));
    input.push_back(std::make_pair(3, 4.0f));
    input.push_back(std::make_pair(5, 6.0f));
    input.push_back(std::make_pair(7, 8.0f));
    BOOST_CHECK_EQUAL(input.size(), size_t(4));

    boost::compute::vector<std::pair<int, float> > output(4);
    boost::compute::copy(input.begin(), input.end(), output.begin());
    BOOST_CHECK(output[0] == std::make_pair(1, 2.0f));
    BOOST_CHECK(output[1] == std::make_pair(3, 4.0f));
    BOOST_CHECK(output[2] == std::make_pair(5, 6.0f));
    BOOST_CHECK(output[3] == std::make_pair(7, 8.0f));
}

BOOST_AUTO_TEST_CASE(fill_pair_vector)
{
    boost::compute::vector<std::pair<int, float> > vector(5);
    boost::compute::fill(vector.begin(), vector.end(), std::make_pair(4, 2.0f));
    BOOST_CHECK(vector[0] == std::make_pair(4, 2.0f));
    BOOST_CHECK(vector[1] == std::make_pair(4, 2.0f));
    BOOST_CHECK(vector[2] == std::make_pair(4, 2.0f));
    BOOST_CHECK(vector[3] == std::make_pair(4, 2.0f));
    BOOST_CHECK(vector[4] == std::make_pair(4, 2.0f));
}

BOOST_AUTO_TEST_CASE(transform_pair_get)
{
    boost::compute::vector<std::pair<int, float> > input;
    input.push_back(std::make_pair(1, 2.0f));
    input.push_back(std::make_pair(3, 4.0f));
    input.push_back(std::make_pair(5, 6.0f));
    input.push_back(std::make_pair(7, 8.0f));

    boost::compute::vector<int> first_output(4);
    boost::compute::transform(
        input.begin(),
        input.end(),
        first_output.begin(),
        ::boost::compute::get_pair<0, int, float>()
    );
    BOOST_CHECK_EQUAL(int(first_output[0]), int(1));
    BOOST_CHECK_EQUAL(int(first_output[1]), int(3));
    BOOST_CHECK_EQUAL(int(first_output[2]), int(5));
    BOOST_CHECK_EQUAL(int(first_output[3]), int(7));

    boost::compute::vector<float> second_output(4);
    boost::compute::transform(
        input.begin(),
        input.end(),
        second_output.begin(),
        ::boost::compute::get_pair<1, int, float>()
    );
    BOOST_CHECK_EQUAL(float(second_output[0]), float(2.0f));
    BOOST_CHECK_EQUAL(float(second_output[1]), float(4.0f));
    BOOST_CHECK_EQUAL(float(second_output[2]), float(6.0f));
    BOOST_CHECK_EQUAL(float(second_output[3]), float(8.0f));
}

BOOST_AUTO_TEST_CASE(find_vector_pair)
{
    boost::compute::vector<std::pair<int, float> > vector;
    vector.push_back(std::make_pair(1, 1.1f));
    vector.push_back(std::make_pair(2, 2.2f));
    vector.push_back(std::make_pair(3, 3.3f));
    BOOST_CHECK_EQUAL(vector.size(), size_t(3));

    BOOST_CHECK(
        boost::compute::find(
            boost::compute::make_transform_iterator(
                vector.begin(),
                boost::compute::get_pair<0, int, float>()
            ),
            boost::compute::make_transform_iterator(
                vector.end(),
                boost::compute::get_pair<0, int, float>()
            ),
            int(2)
        ).base() == vector.begin() + 1
    );

    BOOST_CHECK(
        boost::compute::find(
            boost::compute::make_transform_iterator(
                vector.begin(),
                boost::compute::get_pair<1, int, float>()
            ),
            boost::compute::make_transform_iterator(
                vector.end(),
                boost::compute::get_pair<1, int, float>()
            ),
            float(3.3f)
        ).base() == vector.begin() + 2
    );
}
