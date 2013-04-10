//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestTransform
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(transform_int_abs)
{
    int data[] = { 1, -2, -3, -4, 5 };
    bc::vector<int> vector(data, data + 5);
    BOOST_CHECK_EQUAL(vector[0], 1);
    BOOST_CHECK_EQUAL(vector[1], -2);
    BOOST_CHECK_EQUAL(vector[2], -3);
    BOOST_CHECK_EQUAL(vector[3], -4);
    BOOST_CHECK_EQUAL(vector[4], 5);

    bc::transform(vector.begin(),
                  vector.end(),
                  vector.begin(),
                  bc::abs<int>());
    BOOST_CHECK_EQUAL(vector[0], 1);
    BOOST_CHECK_EQUAL(vector[1], 2);
    BOOST_CHECK_EQUAL(vector[2], 3);
    BOOST_CHECK_EQUAL(vector[3], 4);
    BOOST_CHECK_EQUAL(vector[4], 5);
}

BOOST_AUTO_TEST_CASE(transform_float_sqrt)
{
    float data[] = { 1.0f, 4.0f, 9.0f, 16.0f };
    bc::vector<float> vector(data, data + 4);
    BOOST_CHECK_EQUAL(float(vector[0]), 1.0f);
    BOOST_CHECK_EQUAL(float(vector[1]), 4.0f);
    BOOST_CHECK_EQUAL(float(vector[2]), 9.0f);
    BOOST_CHECK_EQUAL(float(vector[3]), 16.0f);

    bc::transform(vector.begin(),
                  vector.end(),
                  vector.begin(),
                  bc::sqrt<float>());
    BOOST_CHECK_CLOSE(float(vector[0]), 1.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(vector[1]), 2.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(vector[2]), 3.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(vector[3]), 4.0f, 1e-4);
}

BOOST_AUTO_TEST_CASE(transform_float_clamp)
{
    float data[] = { 10.f, 20.f, 30.f, 40.f, 50.f };
    bc::vector<float> vector(data, data + 5);
    BOOST_CHECK_EQUAL(float(vector[0]), 10.f);
    BOOST_CHECK_EQUAL(float(vector[1]), 20.f);
    BOOST_CHECK_EQUAL(float(vector[2]), 30.f);
    BOOST_CHECK_EQUAL(float(vector[3]), 40.f);
    BOOST_CHECK_EQUAL(float(vector[4]), 50.f);

    bc::transform(vector.begin(),
                  vector.end(),
                  vector.begin(),
                  clamp(bc::_1, 15.f, 45.f));

    BOOST_CHECK_EQUAL(float(vector[0]), 15.f);
    BOOST_CHECK_EQUAL(float(vector[1]), 20.f);
    BOOST_CHECK_EQUAL(float(vector[2]), 30.f);
    BOOST_CHECK_EQUAL(float(vector[3]), 40.f);
    BOOST_CHECK_EQUAL(float(vector[4]), 45.f);
}

BOOST_AUTO_TEST_CASE(transform_add_int)
{
    int data1[] = { 1, 2, 3, 4 };
    bc::vector<int> input1(data1, data1 + 4);

    int data2[] = { 10, 20, 30, 40 };
    bc::vector<int> input2(data2, data2 + 4);

    bc::vector<int> output(4);
    bc::transform(input1.begin(),
                  input1.end(),
                  input2.begin(),
                  output.begin(),
                  bc::plus<int>());
    BOOST_CHECK_EQUAL(output[0], 11);
    BOOST_CHECK_EQUAL(output[1], 22);
    BOOST_CHECK_EQUAL(output[2], 33);
    BOOST_CHECK_EQUAL(output[3], 44);

    bc::transform(input1.begin(),
                  input1.end(),
                  input2.begin(),
                  output.begin(),
                  bc::multiplies<int>());
    BOOST_CHECK_EQUAL(output[0], 10);
    BOOST_CHECK_EQUAL(output[1], 40);
    BOOST_CHECK_EQUAL(output[2], 90);
    BOOST_CHECK_EQUAL(output[3], 160);
}

BOOST_AUTO_TEST_CASE(transform_pow4)
{
    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    bc::vector<float> vector(data, data + 4);
    BOOST_CHECK_EQUAL(float(vector[0]), 1.0f);
    BOOST_CHECK_EQUAL(float(vector[1]), 2.0f);
    BOOST_CHECK_EQUAL(float(vector[2]), 3.0f);
    BOOST_CHECK_EQUAL(float(vector[3]), 4.0f);

    bc::vector<float> result(4);
    bc::transform(vector.begin(),
                  vector.end(),
                  result.begin(),
                  pown(bc::_1, 4));
    BOOST_CHECK_CLOSE(float(result[0]), 1.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(result[1]), 16.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(result[2]), 81.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(result[3]), 256.0f, 1e-4);
}

BOOST_AUTO_TEST_CASE(transform_custom_function)
{
    float data[] = { 9.0f, 7.0f, 5.0f, 3.0f };
    bc::vector<float> vector(data, data + 4);

    const char source[] =
        "float pow3add4(float x){ return pow(x, 3.0f) + 4.0f; }";
    bc::function<float (float)> pow3add4 =
        bc::make_function_from_source<float (float)>("pow3add4", source);

    bc::vector<float> result(4);
    bc::transform(vector.begin(),
                  vector.end(),
                  result.begin(),
                  pow3add4);
    BOOST_CHECK_CLOSE(float(result[0]), 733.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(result[1]), 347.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(result[2]), 129.0f, 1e-4);
    BOOST_CHECK_CLOSE(float(result[3]), 31.0f, 1e-4);
}

BOOST_AUTO_TEST_CASE(extract_vector_component)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);

    int data[] = { 1, 2,
                   3, 4,
                   5, 6,
                   7, 8 };
    bc::vector<bc::int2_> vector(reinterpret_cast<bc::int2_ *>(data),
                                 reinterpret_cast<bc::int2_ *>(data) + 4,
                                 context);
    BOOST_CHECK_EQUAL(vector[0], bc::int2_(1, 2));
    BOOST_CHECK_EQUAL(vector[1], bc::int2_(3, 4));
    BOOST_CHECK_EQUAL(vector[2], bc::int2_(5, 6));
    BOOST_CHECK_EQUAL(vector[3], bc::int2_(7, 8));

    bc::vector<int> x_components(4, context);
    bc::transform(vector.begin(),
                  vector.end(),
                  x_components.begin(),
                  bc::vector_component<bc::int2_, 0>());
    BOOST_CHECK_EQUAL(x_components[0], 1);
    BOOST_CHECK_EQUAL(x_components[1], 3);
    BOOST_CHECK_EQUAL(x_components[2], 5);
    BOOST_CHECK_EQUAL(x_components[3], 7);

    bc::vector<int> y_components(4, context);
    bc::transform(vector.begin(),
                  vector.end(),
                  y_components.begin(),
                  bc::vector_component<bc::int2_, 1>());
    BOOST_CHECK_EQUAL(y_components[0], 2);
    BOOST_CHECK_EQUAL(y_components[1], 4);
    BOOST_CHECK_EQUAL(y_components[2], 6);
    BOOST_CHECK_EQUAL(y_components[3], 8);
}

BOOST_AUTO_TEST_CASE(transform_pinned_vector)
{
    int data[] = { 2, -3, 4, -5, 6, -7 };
    std::vector<int> vector(data, data + 6);

    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    bc::buffer buffer(context,
                      vector.size() * sizeof(int),
                      bc::buffer::read_write | bc::buffer::use_host_ptr,
                      &vector[0]);

    bc::transform(bc::make_buffer_iterator<int>(buffer, 0),
                  bc::make_buffer_iterator<int>(buffer, 6),
                  bc::make_buffer_iterator<int>(buffer, 0),
                  bc::abs<int>(),
                  queue);

    void *ptr = queue.enqueue_map_buffer(buffer,
                                         bc::command_queue::map_read,
                                         0,
                                         buffer.size());
    BOOST_VERIFY(ptr == &vector[0]);
    BOOST_CHECK_EQUAL(vector[0], 2);
    BOOST_CHECK_EQUAL(vector[1], 3);
    BOOST_CHECK_EQUAL(vector[2], 4);
    BOOST_CHECK_EQUAL(vector[3], 5);
    BOOST_CHECK_EQUAL(vector[4], 6);
    BOOST_CHECK_EQUAL(vector[5], 7);
    queue.enqueue_unmap_buffer(buffer, ptr);
}

BOOST_AUTO_TEST_CASE(transform_popcount)
{
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    using boost::compute::uint_;

    uint_ data[] = { 0, 1, 2, 3, 4, 45, 127, 5000, 789, 15963 };
    bc::vector<uint_> input(data, data + 10, context);
    bc::vector<uint_> output(input.size(), context);

    bc::transform(
        input.begin(),
        input.end(),
        output.begin(),
        bc::popcount<uint_>(),
        queue
    );
    queue.finish();

    BOOST_CHECK_EQUAL(uint_(output[0]), uint_(0));
    BOOST_CHECK_EQUAL(uint_(output[1]), uint_(1));
    BOOST_CHECK_EQUAL(uint_(output[2]), uint_(1));
    BOOST_CHECK_EQUAL(uint_(output[3]), uint_(2));
    BOOST_CHECK_EQUAL(uint_(output[4]), uint_(1));
    BOOST_CHECK_EQUAL(uint_(output[5]), uint_(4));
    BOOST_CHECK_EQUAL(uint_(output[6]), uint_(7));
    BOOST_CHECK_EQUAL(uint_(output[7]), uint_(5));
    BOOST_CHECK_EQUAL(uint_(output[8]), uint_(5));
    BOOST_CHECK_EQUAL(uint_(output[9]), uint_(10));
}

// generates the first 25 fibonacci numbers in parallel using the
// rounding-based fibonacci formula
BOOST_AUTO_TEST_CASE(generate_fibonacci_sequence)
{
    using boost::compute::uint_;

    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    boost::compute::vector<uint_> sequence(25, context);

    const char nth_fibonacci_source[] =
        "inline uint nth_fibonacci(const uint n)\n"
        "{\n"
        "    const float golden_ratio = (1.f + sqrt(5.f)) / 2.f;\n"
        "    return floor(pown(golden_ratio, n) / sqrt(5.f) + 0.5f);\n"
        "}\n";

    boost::compute::function<uint_(const uint_)> nth_fibonacci =
        boost::compute::make_function_from_source<uint_(const uint_)>(
            "nth_fibonacci", nth_fibonacci_source);

    boost::compute::transform(
        boost::compute::make_counting_iterator(uint_(0)),
        boost::compute::make_counting_iterator(uint_(sequence.size())),
        sequence.begin(),
        nth_fibonacci,
        queue
    );
    queue.finish();

    BOOST_CHECK_EQUAL(uint_(sequence[0]), 0);
    BOOST_CHECK_EQUAL(uint_(sequence[1]), 1);
    BOOST_CHECK_EQUAL(uint_(sequence[2]), 1);
    BOOST_CHECK_EQUAL(uint_(sequence[5]), 5);
    BOOST_CHECK_EQUAL(uint_(sequence[9]), 34);
    BOOST_CHECK_EQUAL(uint_(sequence[15]), 610);
    BOOST_CHECK_EQUAL(uint_(sequence[24]), 46368);
}
