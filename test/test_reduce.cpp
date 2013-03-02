//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestReduce
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/constant_iterator.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(reduce_int)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    int data[] = { 1, 5, 9, 13, 17 };
    bc::vector<int> vector(data, data + 5, context);
    int sum = bc::reduce(vector.begin(),
                         vector.end(),
                         0,
                         bc::plus<int>(),
                         queue);
    BOOST_CHECK_EQUAL(sum, 45);

    int product = bc::reduce(vector.begin(),
                             vector.end(),
                             1,
                             bc::multiplies<int>(),
                             queue);
    BOOST_CHECK_EQUAL(product, 9945);
}

BOOST_AUTO_TEST_CASE(reduce_int_min_max)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    int data[] = { 11, 5, 92, 13, 42 };
    bc::vector<int> vector(data, data + 5, context);
    BOOST_CHECK_EQUAL(
        bc::reduce(vector.begin(),
                   vector.end(),
                   (std::numeric_limits<int>::max)(),
                   bc::min<int>(),
                   queue
        ),
        5
    );

    BOOST_CHECK_EQUAL(
        bc::reduce(vector.begin(),
                   vector.end(),
                   (std::numeric_limits<int>::min)(),
                   bc::max<int>(),
                   queue
        ),
        92
    );
}

BOOST_AUTO_TEST_CASE(reduce_int2)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    std::vector<bc::int2_> data;
    for(int i = 0; i < 6; i++){
        bc::int2_ value;
        value[0] = i + 1;
        value[1] = 2 * i + 1;
        data.push_back(value);
    }

    bc::vector<bc::int2_> vector(data.begin(), data.end(), context);
    bc::int2_ sum = bc::reduce(vector.begin(),
                               vector.end(),
                               bc::int2_(0, 0),
                               bc::plus<bc::int2_>(),
                               queue);
    BOOST_CHECK_EQUAL(sum, bc::int2_(21, 36));
}

BOOST_AUTO_TEST_CASE(reduce_pinned_vector)
{
    int data[] = { 2, 5, 8, 11, 15 };
    std::vector<int> vector(data, data + 5);

    bc::buffer buffer(bc::system::default_context(),
                      vector.size() * sizeof(int),
                      bc::buffer::read_only | bc::buffer::use_host_ptr,
                      &vector[0]);

    int sum = bc::reduce(bc::make_buffer_iterator<int>(buffer, 0),
                         bc::make_buffer_iterator<int>(buffer, 5),
                         0,
                         bc::plus<int>());
    BOOST_CHECK_EQUAL(sum, 41);
}

BOOST_AUTO_TEST_CASE(reduce_constant_iterator)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_constant_iterator(1, 0),
                   bc::make_constant_iterator(1, 5),
                   0,
                   bc::plus<int>(),
                   queue),
        int(5)
    );
    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_constant_iterator(3, 0),
                   bc::make_constant_iterator(3, 5),
                   0,
                   bc::plus<int>(),
                   queue),
        int(15)
    );
    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_constant_iterator(2, 0),
                   bc::make_constant_iterator(2, 5),
                   1,
                   bc::multiplies<int>(),
                   queue),
        int(32)
    );
}

BOOST_AUTO_TEST_CASE(reduce_counting_iterator)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_counting_iterator(1),
                   bc::make_counting_iterator(10),
                   0,
                   bc::plus<int>(),
                   queue),
        int(45)
    );
    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_counting_iterator(1),
                   bc::make_counting_iterator(5),
                   1,
                   bc::multiplies<int>(),
                   queue),
        int(24)
    );
}

BOOST_AUTO_TEST_CASE(reduce_transform_iterator)
{
    using ::boost::compute::_1;

    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    int data[] = { 1, 3, 5, 7, 9 };
    bc::vector<int> vector(data, data + 5, context);

    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_transform_iterator(vector.begin(), _1 + 1),
                   bc::make_transform_iterator(vector.end(), _1 + 1),
                   0,
                   bc::plus<int>(),
                   queue),
        int(30)
    );

    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_transform_iterator(vector.begin(), _1 > 4),
                   bc::make_transform_iterator(vector.end(), _1 > 4),
                   0,
                   bc::plus<int>(),
                   queue),
        int(3)
    );

    BOOST_CHECK_EQUAL(
        bc::reduce(bc::make_transform_iterator(vector.begin(), _1 * _1),
                   bc::make_transform_iterator(vector.end(), _1 * _1),
                   0,
                   bc::plus<int>(),
                   queue),
        int(165)
    );
}
