//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestInsertionSort
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/is_sorted.hpp>
#include <boost/compute/algorithm/detail/insertion_sort.hpp>
#include <boost/compute/container/vector.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(sort_char_vector)
{
    using boost::compute::char_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    char_ data[] = { 'c', 'a', '0', '7', 'B', 'F', '\0', '$' };
    boost::compute::vector<char_> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], '\0');
    BOOST_CHECK_EQUAL(data[1], '$');
    BOOST_CHECK_EQUAL(data[2], '0');
    BOOST_CHECK_EQUAL(data[3], '7');
    BOOST_CHECK_EQUAL(data[4], 'B');
    BOOST_CHECK_EQUAL(data[5], 'F');
    BOOST_CHECK_EQUAL(data[6], 'a');
    BOOST_CHECK_EQUAL(data[7], 'c');
}

BOOST_AUTO_TEST_CASE(sort_uchar_vector)
{
    using boost::compute::uchar_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    uchar_ data[] = { 0x12, 0x00, 0xFF, 0xB4, 0x80, 0x32, 0x64, 0xA2 };
    boost::compute::vector<uchar_> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], uchar_(0x00));
    BOOST_CHECK_EQUAL(data[1], uchar_(0x12));
    BOOST_CHECK_EQUAL(data[2], uchar_(0x32));
    BOOST_CHECK_EQUAL(data[3], uchar_(0x64));
    BOOST_CHECK_EQUAL(data[4], uchar_(0x80));
    BOOST_CHECK_EQUAL(data[5], uchar_(0xA2));
    BOOST_CHECK_EQUAL(data[6], uchar_(0xB4));
    BOOST_CHECK_EQUAL(data[7], uchar_(0xFF));
}

BOOST_AUTO_TEST_CASE(sort_short_vector)
{
    using boost::compute::short_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    short_ data[] = { -4, 152, -94, 963, 31002, -456, 0, -2113 };
    boost::compute::vector<short_> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], short_(-2113));
    BOOST_CHECK_EQUAL(data[1], short_(-456));
    BOOST_CHECK_EQUAL(data[2], short_(-94));
    BOOST_CHECK_EQUAL(data[3], short_(-4));
    BOOST_CHECK_EQUAL(data[4], short_(0));
    BOOST_CHECK_EQUAL(data[5], short_(152));
    BOOST_CHECK_EQUAL(data[6], short_(963));
    BOOST_CHECK_EQUAL(data[7], short_(31002));
}

BOOST_AUTO_TEST_CASE(sort_ushort_vector)
{
    using boost::compute::ushort_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    ushort_ data[] = { 4, 152, 94, 963, 63202, 34560, 0, 2113 };
    boost::compute::vector<ushort_> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], ushort_(0));
    BOOST_CHECK_EQUAL(data[1], ushort_(4));
    BOOST_CHECK_EQUAL(data[2], ushort_(94));
    BOOST_CHECK_EQUAL(data[3], ushort_(152));
    BOOST_CHECK_EQUAL(data[4], ushort_(963));
    BOOST_CHECK_EQUAL(data[5], ushort_(2113));
    BOOST_CHECK_EQUAL(data[6], ushort_(34560));
    BOOST_CHECK_EQUAL(data[7], ushort_(63202));
}

BOOST_AUTO_TEST_CASE(sort_int_vector)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    int data[] = { -4, 152, -5000, 963, 75321, -456, 0, 1112 };
    boost::compute::vector<int> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], -5000);
    BOOST_CHECK_EQUAL(data[1], -456);
    BOOST_CHECK_EQUAL(data[2], -4);
    BOOST_CHECK_EQUAL(data[3], 0);
    BOOST_CHECK_EQUAL(data[4], 152);
    BOOST_CHECK_EQUAL(data[5], 963);
    BOOST_CHECK_EQUAL(data[6], 1112);
    BOOST_CHECK_EQUAL(data[7], 75321);
}

BOOST_AUTO_TEST_CASE(sort_uint_vector)
{
    using boost::compute::uint_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    uint_ data[] = { 500, 1988, 123456, 562, 0, 4000000, 9852, 102030 };
    boost::compute::vector<uint_> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], uint_(0));
    BOOST_CHECK_EQUAL(data[1], uint_(500));
    BOOST_CHECK_EQUAL(data[2], uint_(562));
    BOOST_CHECK_EQUAL(data[3], uint_(1988));
    BOOST_CHECK_EQUAL(data[4], uint_(9852));
    BOOST_CHECK_EQUAL(data[5], uint_(102030));
    BOOST_CHECK_EQUAL(data[6], uint_(123456));
    BOOST_CHECK_EQUAL(data[7], uint_(4000000));
}

BOOST_AUTO_TEST_CASE(sort_long_vector)
{
    using boost::compute::long_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    long_ data[] = { 500, 1988, 123456, 562, 0, 4000000, 9852, 102030 };
    boost::compute::vector<long_> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], long_(0));
    BOOST_CHECK_EQUAL(data[1], long_(500));
    BOOST_CHECK_EQUAL(data[2], long_(562));
    BOOST_CHECK_EQUAL(data[3], long_(1988));
    BOOST_CHECK_EQUAL(data[4], long_(9852));
    BOOST_CHECK_EQUAL(data[5], long_(102030));
    BOOST_CHECK_EQUAL(data[6], long_(123456));
    BOOST_CHECK_EQUAL(data[7], long_(4000000));
}

BOOST_AUTO_TEST_CASE(sort_ulong_vector)
{
    using boost::compute::ulong_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    ulong_ data[] = { 500, 1988, 123456, 562, 0, 4000000, 9852, 102030 };
    boost::compute::vector<ulong_> vector(data, data + 8, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(8));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], ulong_(0));
    BOOST_CHECK_EQUAL(data[1], ulong_(500));
    BOOST_CHECK_EQUAL(data[2], ulong_(562));
    BOOST_CHECK_EQUAL(data[3], ulong_(1988));
    BOOST_CHECK_EQUAL(data[4], ulong_(9852));
    BOOST_CHECK_EQUAL(data[5], ulong_(102030));
    BOOST_CHECK_EQUAL(data[6], ulong_(123456));
    BOOST_CHECK_EQUAL(data[7], ulong_(4000000));
}

BOOST_AUTO_TEST_CASE(sort_float_vector)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    float data[] = { -6023.0f, 152.5f, -63.0f, 1234567.0f, 11.2f,
                     -5000.1f, 0.0f, 14.0f, -8.25f, -0.0f };
    boost::compute::vector<float> vector(data, data + 10, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(10));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], -6023.0f);
    BOOST_CHECK_EQUAL(data[1], -5000.1f);
    BOOST_CHECK_EQUAL(data[2], -63.0f);
    BOOST_CHECK_EQUAL(data[3], -8.25f);
    BOOST_CHECK_EQUAL(data[4], -0.0f);
    BOOST_CHECK_EQUAL(data[5], 0.0f);
    BOOST_CHECK_EQUAL(data[6], 11.2f);
    BOOST_CHECK_EQUAL(data[7], 14.0f);
    BOOST_CHECK_EQUAL(data[8], 152.5f);
    BOOST_CHECK_EQUAL(data[9], 1234567.0f);
}

BOOST_AUTO_TEST_CASE(sort_double_vector)
{
    boost::compute::device device = boost::compute::system::default_device();
    if(!device.supports_extension("cl_khr_fp64")){
        std::cout << "skipping test: device does not support double" << std::endl;
        return;
    }

    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    double data[] = { -6023.0, 152.5, -63.0, 1234567.0, 11.2,
                     -5000.1, 0.0, 14.0, -8.25, -0.0 };
    boost::compute::vector<double> vector(data, data + 10, context);
    BOOST_CHECK_EQUAL(vector.size(), size_t(10));
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == false);

    boost::compute::detail::serial_insertion_sort(vector.begin(), vector.end(), queue);
    BOOST_CHECK(boost::compute::is_sorted(vector.begin(), vector.end(), queue) == true);

    boost::compute::copy(vector.begin(), vector.end(), data, queue);
    BOOST_CHECK_EQUAL(data[0], -6023.0);
    BOOST_CHECK_EQUAL(data[1], -5000.1);
    BOOST_CHECK_EQUAL(data[2], -63.0);
    BOOST_CHECK_EQUAL(data[3], -8.25);
    BOOST_CHECK_EQUAL(data[4], -0.0);
    BOOST_CHECK_EQUAL(data[5], 0.0);
    BOOST_CHECK_EQUAL(data[6], 11.2);
    BOOST_CHECK_EQUAL(data[7], 14.0);
    BOOST_CHECK_EQUAL(data[8], 152.5);
    BOOST_CHECK_EQUAL(data[9], 1234567.0);
}
