//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestCopy
#include <boost/test/unit_test.hpp>

#include <list>
#include <vector>

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/copy_n.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/detail/swizzle_iterator.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(copy_on_device)
{
    float data[] = { 6.1f, 10.2f, 19.3f, 25.4f };
    bc::vector<float> a(4);
    bc::copy(data, data + 4, a.begin());
    BOOST_CHECK_EQUAL(a[0], 6.1f);
    BOOST_CHECK_EQUAL(a[1], 10.2f);
    BOOST_CHECK_EQUAL(a[2], 19.3f);
    BOOST_CHECK_EQUAL(a[3], 25.4f);

    bc::vector<float> b(4);
    bc::fill(b.begin(), b.end(), 0);
    BOOST_CHECK_EQUAL(b[0], 0.0f);
    BOOST_CHECK_EQUAL(b[1], 0.0f);
    BOOST_CHECK_EQUAL(b[2], 0.0f);
    BOOST_CHECK_EQUAL(b[3], 0.0f);

    bc::copy(a.begin(), a.end(), b.begin());
    BOOST_CHECK_EQUAL(b[0], 6.1f);
    BOOST_CHECK_EQUAL(b[1], 10.2f);
    BOOST_CHECK_EQUAL(b[2], 19.3f);
    BOOST_CHECK_EQUAL(b[3], 25.4f);
}

BOOST_AUTO_TEST_CASE(copy)
{
    int data[] = { 1, 2, 5, 6 };
    bc::vector<int> vector(4);
    bc::copy(data, data + 4, vector.begin());
    BOOST_CHECK_EQUAL(vector[0], 1);
    BOOST_CHECK_EQUAL(vector[1], 2);
    BOOST_CHECK_EQUAL(vector[2], 5);
    BOOST_CHECK_EQUAL(vector[3], 6);

    std::vector<int> host_vector(4);
    bc::copy(vector.begin(), vector.end(), host_vector.begin());
    BOOST_CHECK_EQUAL(host_vector[0], 1);
    BOOST_CHECK_EQUAL(host_vector[1], 2);
    BOOST_CHECK_EQUAL(host_vector[2], 5);
    BOOST_CHECK_EQUAL(host_vector[3], 6);
}

// Test copying from a std::list to a bc::vector. This differs from
// the test copying from std::vector because std::list has non-contigous
// storage for its data values.
BOOST_AUTO_TEST_CASE(copy_from_host_list)
{
    int data[] = { -4, 12, 9, 0 };
    std::list<int> host_list(data, data + 4);

    bc::vector<int> vector(4);
    bc::copy(host_list.begin(), host_list.end(), vector.begin());
    BOOST_CHECK_EQUAL(vector[0], -4);
    BOOST_CHECK_EQUAL(vector[1], 12);
    BOOST_CHECK_EQUAL(vector[2], 9);
    BOOST_CHECK_EQUAL(vector[3], 0);
}

BOOST_AUTO_TEST_CASE(copy_n_int)
{
    int data[] = { 1, 2, 3, 4, 5 };
    bc::vector<int> a(data, data + 5);

    bc::vector<int> b(5);
    bc::fill(b.begin(), b.end(), 0);
    bc::copy_n(a.begin(), 3, b.begin());
    BOOST_CHECK_EQUAL(b[0], 1);
    BOOST_CHECK_EQUAL(b[1], 2);
    BOOST_CHECK_EQUAL(b[2], 3);
    BOOST_CHECK_EQUAL(b[3], 0);
    BOOST_CHECK_EQUAL(b[4], 0);

    bc::copy_n(b.begin(), 4, a.begin());
    BOOST_CHECK_EQUAL(a[0], 1);
    BOOST_CHECK_EQUAL(a[1], 2);
    BOOST_CHECK_EQUAL(a[2], 3);
    BOOST_CHECK_EQUAL(a[3], 0);
    BOOST_CHECK_EQUAL(a[4], 5);
}

BOOST_AUTO_TEST_CASE(copy_swizzle_iterator)
{
    using bc::int2_;
    using bc::int4_;

    int data[] = { 1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 1, 2, 3,
                   4, 5, 6, 7 };

    bc::vector<int4_> input(reinterpret_cast<int4_*>(data),
                            reinterpret_cast<int4_*>(data) + 4);
    BOOST_CHECK_EQUAL(input.size(), size_t(4));
    BOOST_CHECK_EQUAL(input[0], int4_(1, 2, 3, 4));
    BOOST_CHECK_EQUAL(input[1], int4_(5, 6, 7, 8));
    BOOST_CHECK_EQUAL(input[2], int4_(9, 1, 2, 3));
    BOOST_CHECK_EQUAL(input[3], int4_(4, 5, 6, 7));

    bc::vector<int4_> output4(4);
    bc::copy(
        bc::detail::make_swizzle_iterator<4>(input.begin(), "wzyx"),
        bc::detail::make_swizzle_iterator<4>(input.end(), "wzyx"),
        output4.begin()
    );
    BOOST_CHECK_EQUAL(output4[0], int4_(4, 3, 2, 1));
    BOOST_CHECK_EQUAL(output4[1], int4_(8, 7, 6, 5));
    BOOST_CHECK_EQUAL(output4[2], int4_(3, 2, 1, 9));
    BOOST_CHECK_EQUAL(output4[3], int4_(7, 6, 5, 4));

    bc::vector<int2_> output2(4);
    bc::copy(
        bc::detail::make_swizzle_iterator<2>(input.begin(), "xz"),
        bc::detail::make_swizzle_iterator<2>(input.end(), "xz"),
        output2.begin()
    );
    BOOST_CHECK_EQUAL(output2[0], int2_(1, 3));
    BOOST_CHECK_EQUAL(output2[1], int2_(5, 7));
    BOOST_CHECK_EQUAL(output2[2], int2_(9, 2));
    BOOST_CHECK_EQUAL(output2[3], int2_(4, 6));

    bc::vector<int> output1(4);
    bc::copy(
        bc::detail::make_swizzle_iterator<1>(input.begin(), "y"),
        bc::detail::make_swizzle_iterator<1>(input.end(), "y"),
        output1.begin()
    );
    BOOST_CHECK_EQUAL(output1[0], int(2));
    BOOST_CHECK_EQUAL(output1[1], int(6));
    BOOST_CHECK_EQUAL(output1[2], int(1));
    BOOST_CHECK_EQUAL(output1[3], int(5));
}

BOOST_AUTO_TEST_CASE(copy_int_async)
{
    // setup context and queue
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    // setup host data
    int host_data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    typedef int* host_iterator;

    // setup device data
    bc::vector<int> device_data(8, context);
    typedef bc::vector<int>::iterator device_iterator;

    // copy data to device
    bc::future<device_iterator> host_to_device_future =
        bc::copy_async(host_data, host_data + 8, device_data.begin(), queue);

    // wait for copy to complete
    host_to_device_future.wait();

    // check results
    BOOST_CHECK_EQUAL(device_data[0], int(1));
    BOOST_CHECK_EQUAL(device_data[1], int(2));
    BOOST_CHECK_EQUAL(device_data[2], int(3));
    BOOST_CHECK_EQUAL(device_data[3], int(4));
    BOOST_CHECK_EQUAL(device_data[4], int(5));
    BOOST_CHECK_EQUAL(device_data[5], int(6));
    BOOST_CHECK_EQUAL(device_data[6], int(7));
    BOOST_CHECK_EQUAL(device_data[7], int(8));
    BOOST_VERIFY(host_to_device_future.get() == device_data.end());

    // fill host data with zeros
    std::fill(host_data, host_data + 8, int(0));

    // copy data back to host
    bc::future<host_iterator> device_to_host_future =
        bc::copy_async(device_data.begin(), device_data.end(), host_data, queue);

    // wait for copy to complete
    device_to_host_future.wait();

    // check results
    BOOST_CHECK_EQUAL(host_data[0], int(1));
    BOOST_CHECK_EQUAL(host_data[1], int(2));
    BOOST_CHECK_EQUAL(host_data[2], int(3));
    BOOST_CHECK_EQUAL(host_data[3], int(4));
    BOOST_CHECK_EQUAL(host_data[4], int(5));
    BOOST_CHECK_EQUAL(host_data[5], int(6));
    BOOST_CHECK_EQUAL(host_data[6], int(7));
    BOOST_CHECK_EQUAL(host_data[7], int(8));
    BOOST_VERIFY(device_to_host_future.get() == host_data + 8);
}
