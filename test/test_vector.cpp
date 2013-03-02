//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestVector
#include <boost/test/unit_test.hpp>

#include <boost/concept_check.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/algorithm/remove.hpp>
#include <boost/compute/container/vector.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(concept_check)
{
    BOOST_CONCEPT_ASSERT((boost::Container<bc::vector<int> >));
    BOOST_CONCEPT_ASSERT((boost::SequenceConcept<bc::vector<int> >));
    BOOST_CONCEPT_ASSERT((boost::ReversibleContainer<bc::vector<int> >));
    BOOST_CONCEPT_ASSERT((boost::RandomAccessIterator<bc::vector<int>::iterator>));
    BOOST_CONCEPT_ASSERT((boost::RandomAccessIterator<bc::vector<int>::const_iterator>));
}

BOOST_AUTO_TEST_CASE(size)
{
    bc::vector<int> empty_vector;
    BOOST_CHECK_EQUAL(empty_vector.size(), size_t(0));
    BOOST_CHECK_EQUAL(empty_vector.empty(), true);

    bc::vector<int> int_vector(10);
    BOOST_CHECK_EQUAL(int_vector.size(), size_t(10));
    BOOST_CHECK_EQUAL(int_vector.empty(), false);
}

BOOST_AUTO_TEST_CASE(resize)
{
    bc::vector<int> int_vector(10);
    BOOST_CHECK_EQUAL(int_vector.size(), size_t(10));

    int_vector.resize(20);
    BOOST_CHECK_EQUAL(int_vector.size(), size_t(20));

    int_vector.resize(5);
    BOOST_CHECK_EQUAL(int_vector.size(), size_t(5));
}

BOOST_AUTO_TEST_CASE(array_operator)
{
    bc::vector<int> vector(10);
    bc::fill(vector.begin(), vector.end(), 0);
    BOOST_CHECK_EQUAL(vector[0], 0);
    BOOST_CHECK_EQUAL(vector[9], 0);

    bc::fill(vector.begin(), vector.end(), 42);
    BOOST_CHECK_EQUAL(vector[0], 42);
    BOOST_CHECK_EQUAL(vector[9], 42);

    vector[0] = 9;
    BOOST_CHECK_EQUAL(vector[0], 9);
}

BOOST_AUTO_TEST_CASE(front_and_back)
{
    int int_data[] = { 1, 2, 3, 4, 5 };
    bc::vector<int> int_vector(5);
    bc::copy(int_data, int_data + 5, int_vector.begin());
    BOOST_CHECK_EQUAL(int_vector.front(), 1);
    BOOST_CHECK_EQUAL(int_vector.back(), 5);

    bc::fill(int_vector.begin(), int_vector.end(), 10);
    BOOST_CHECK_EQUAL(int_vector.front(), 10);
    BOOST_CHECK_EQUAL(int_vector.back(), 10);

    float float_data[] = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f };
    bc::vector<float> float_vector(5);
    bc::copy(float_data, float_data + 5, float_vector.begin());
    BOOST_CHECK_EQUAL(float_vector.front(), 1.1f);
    BOOST_CHECK_EQUAL(float_vector.back(), 5.5f);
}

BOOST_AUTO_TEST_CASE(host_iterator_constructor)
{
    std::vector<int> host_vector;
    host_vector.push_back(10);
    host_vector.push_back(20);
    host_vector.push_back(30);
    host_vector.push_back(40);

    bc::vector<int> device_vector(host_vector.begin(), host_vector.end());
    BOOST_CHECK_EQUAL(device_vector[0], 10);
    BOOST_CHECK_EQUAL(device_vector[1], 20);
    BOOST_CHECK_EQUAL(device_vector[2], 30);
    BOOST_CHECK_EQUAL(device_vector[3], 40);
}

BOOST_AUTO_TEST_CASE(device_iterator_constructor)
{
    int data[] = { 1, 5, 10, 15 };
    bc::vector<int> a(data, data + 4);
    BOOST_CHECK_EQUAL(a[0], 1);
    BOOST_CHECK_EQUAL(a[1], 5);
    BOOST_CHECK_EQUAL(a[2], 10);
    BOOST_CHECK_EQUAL(a[3], 15);

    bc::vector<int> b(a.begin(), a.end());
    BOOST_CHECK_EQUAL(b[0], 1);
    BOOST_CHECK_EQUAL(b[1], 5);
    BOOST_CHECK_EQUAL(b[2], 10);
    BOOST_CHECK_EQUAL(b[3], 15);
}

BOOST_AUTO_TEST_CASE(push_back)
{
    bc::vector<int> vector;
    BOOST_VERIFY(vector.empty());

    vector.push_back(12);
    BOOST_VERIFY(!vector.empty());
    BOOST_CHECK_EQUAL(vector.size(), size_t(1));
    BOOST_CHECK_EQUAL(vector[0], 12);

    vector.push_back(24);
    BOOST_CHECK_EQUAL(vector.size(), size_t(2));
    BOOST_CHECK_EQUAL(vector[0], 12);
    BOOST_CHECK_EQUAL(vector[1], 24);

    vector.push_back(36);
    BOOST_CHECK_EQUAL(vector.size(), size_t(3));
    BOOST_CHECK_EQUAL(vector[0], 12);
    BOOST_CHECK_EQUAL(vector[1], 24);
    BOOST_CHECK_EQUAL(vector[2], 36);

    for(int i = 0; i < 100; i++){
        vector.push_back(i);
    }
    BOOST_CHECK_EQUAL(vector.size(), size_t(103));
    BOOST_CHECK_EQUAL(vector[0], 12);
    BOOST_CHECK_EQUAL(vector[1], 24);
    BOOST_CHECK_EQUAL(vector[2], 36);
    BOOST_CHECK_EQUAL(vector[102], 99);
}

BOOST_AUTO_TEST_CASE(std_sort_float_vector)
{
    std::vector<float> host_vector(10);
    std::generate(host_vector.begin(), host_vector.end(), rand);

    bc::vector<float> vector = host_vector;
    std::sort(vector.begin(), vector.end());
    BOOST_CHECK_LE(vector.front(), vector.back());
}

BOOST_AUTO_TEST_CASE(std_max_element_int_vector)
{
    bc::vector<int> vector;
    vector.push_back(1);
    vector.push_back(9);
    vector.push_back(5);
    vector.push_back(2);
    vector.push_back(6);
    bc::vector<int>::const_iterator iter =
        std::max_element(vector.begin(), vector.end());
    BOOST_VERIFY(iter == vector.begin() + 1);
    BOOST_CHECK_EQUAL(*iter, 9);
}

BOOST_AUTO_TEST_CASE(at)
{
    bc::vector<int> vector;
    vector.push_back(1);
    vector.push_back(2);
    vector.push_back(3);
    BOOST_CHECK_EQUAL(vector.at(0), 1);
    BOOST_CHECK_EQUAL(vector.at(1), 2);
    BOOST_CHECK_EQUAL(vector.at(2), 3);
    BOOST_CHECK_THROW(vector.at(3), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(erase)
{
    int data[] = { 1, 2, 5, 7, 9 };
    bc::vector<int> vector(data, data + 5);
    BOOST_CHECK_EQUAL(vector.size(), 5);

    vector.erase(vector.begin() + 1);
    BOOST_CHECK_EQUAL(vector.size(), 4);
    BOOST_CHECK_EQUAL(vector[0], 1);
    BOOST_CHECK_EQUAL(vector[1], 5);
    BOOST_CHECK_EQUAL(vector[2], 7);
    BOOST_CHECK_EQUAL(vector[3], 9);

    vector.erase(vector.begin() + 2, vector.end());
    BOOST_CHECK_EQUAL(vector.size(), 2);
    BOOST_CHECK_EQUAL(vector[0], 1);
    BOOST_CHECK_EQUAL(vector[1], 5);
}

BOOST_AUTO_TEST_CASE(max_size)
{
    bc::vector<int> vector(100);
    BOOST_CHECK_EQUAL(vector.size(), size_t(100));
    BOOST_VERIFY(vector.max_size() > vector.size());
}

#if !defined(BOOST_NO_RVALUE_REFERENCES)
BOOST_AUTO_TEST_CASE(move_ctor)
{
      int data[] = { 11, 12, 13, 14 };
      bc::vector<int> a(data, data + 4);
      BOOST_CHECK_EQUAL(a.size(), size_t(4));
      BOOST_CHECK_EQUAL(a[0], 11);
      BOOST_CHECK_EQUAL(a[1], 12);
      BOOST_CHECK_EQUAL(a[2], 13);
      BOOST_CHECK_EQUAL(a[3], 14);

      bc::vector<int> b(std::move(a));
      BOOST_CHECK_EQUAL(b.size(), size_t(4));
      BOOST_CHECK_EQUAL(b[0], 11);
      BOOST_CHECK_EQUAL(b[1], 12);
      BOOST_CHECK_EQUAL(b[2], 13);
      BOOST_CHECK_EQUAL(b[3], 14);
}
#endif // !defined(BOOST_NO_RVALUE_REFERENCES)

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && \
    !defined(BOOST_NO_0X_HDR_INITIALIZER_LIST)
BOOST_AUTO_TEST_CASE(initializer_list_ctor)
{
    bc::vector<int> vector = { 2, 4, 6, 8 };
    BOOST_CHECK_EQUAL(vector.size(), size_t(4));
    BOOST_CHECK_EQUAL(vector[0], 2);
    BOOST_CHECK_EQUAL(vector[1], 4);
    BOOST_CHECK_EQUAL(vector[2], 6);
    BOOST_CHECK_EQUAL(vector[3], 8);
}
#endif // !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST)

BOOST_AUTO_TEST_CASE(vector_double)
{
    bc::device device = bc::system::default_device();

    if(!device.supports_extension("cl_khr_fp64")){
        return;
    }

    bc::vector<double> vector;
    vector.push_back(1.21);
    vector.push_back(3.14);
    vector.push_back(7.89);
    BOOST_CHECK_EQUAL(vector.size(), size_t(3));
    BOOST_CHECK_EQUAL(vector[0], 1.21);
    BOOST_CHECK_EQUAL(vector[1], 3.14);
    BOOST_CHECK_EQUAL(vector[2], 7.89);

    bc::vector<double> other = vector;
    BOOST_CHECK_EQUAL(other[0], 1.21);
    BOOST_CHECK_EQUAL(other[1], 3.14);
    BOOST_CHECK_EQUAL(other[2], 7.89);

    bc::fill(other.begin(), other.end(), 8.95);
    BOOST_CHECK_EQUAL(other[0], 8.95);
    BOOST_CHECK_EQUAL(other[1], 8.95);
    BOOST_CHECK_EQUAL(other[2], 8.95);
}

BOOST_AUTO_TEST_CASE(vector_iterator)
{
    bc::vector<int> vector;
    vector.push_back(2);
    vector.push_back(4);
    vector.push_back(6);
    vector.push_back(8);
    BOOST_CHECK_EQUAL(vector.size(), size_t(4));
    BOOST_CHECK_EQUAL(vector[0], 2);
    BOOST_CHECK_EQUAL(*vector.begin(), 2);
    BOOST_CHECK_EQUAL(vector.begin()[0], 2);
    BOOST_CHECK_EQUAL(vector[1], 4);
    BOOST_CHECK_EQUAL(*(vector.begin()+1), 4);
    BOOST_CHECK_EQUAL(vector.begin()[1], 4);
    BOOST_CHECK_EQUAL(vector[2], 6);
    BOOST_CHECK_EQUAL(*(vector.begin()+2), 6);
    BOOST_CHECK_EQUAL(vector.begin()[2], 6);
    BOOST_CHECK_EQUAL(vector[3], 8);
    BOOST_CHECK_EQUAL(*(vector.begin()+3), 8);
    BOOST_CHECK_EQUAL(vector.begin()[3], 8);
}

BOOST_AUTO_TEST_CASE(vector_erase_remove)
{
    int data[] = { 2, 6, 3, 4, 2, 4, 5, 6, 1 };
    bc::vector<int> vector(data, data + 9);
    BOOST_CHECK_EQUAL(vector.size(), size_t(9));

    // remove 4's
    vector.erase(bc::remove(vector.begin(), vector.end(), 4), vector.end());
    BOOST_CHECK_EQUAL(vector.size(), size_t(7));
    BOOST_VERIFY(bc::find(vector.begin(), vector.end(), 4) == vector.end());

    // remove 2's
    vector.erase(bc::remove(vector.begin(), vector.end(), 2), vector.end());
    BOOST_CHECK_EQUAL(vector.size(), size_t(5));
    BOOST_VERIFY(bc::find(vector.begin(), vector.end(), 2) == vector.end());

    // remove 6's
    vector.erase(bc::remove(vector.begin(), vector.end(), 6), vector.end());
    BOOST_CHECK_EQUAL(vector.size(), size_t(3));
    BOOST_VERIFY(bc::find(vector.begin(), vector.end(), 6) == vector.end());

    // check the rest of the values
    BOOST_CHECK_EQUAL(vector[0], 3);
    BOOST_CHECK_EQUAL(vector[1], 5);
    BOOST_CHECK_EQUAL(vector[2], 1);
}
