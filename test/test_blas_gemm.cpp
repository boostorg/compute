//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestBlasGemm
#include <boost/test/unit_test.hpp>

#include <boost/compute/malloc.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/blas/gemm.hpp>

BOOST_AUTO_TEST_CASE(gemm_float3x3)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    float a[9] = { 1.0f, 2.0f, 3.0f,
                   4.0f, 5.0f, 6.0f,
                   7.0f, 8.0f, 9.0f };
    float b[9] = { 10.0f, 12.0f, 14.0f,
                   24.0f, 25.0f, 26.0f,
                   37.0f, 38.0f, 39.0f };
    float c[9] = { 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f };

    boost::compute::device_ptr<float> A =
        boost::compute::malloc<float>(9, context);
    boost::compute::device_ptr<float> B =
        boost::compute::malloc<float>(9, context);
    boost::compute::device_ptr<float> C =
        boost::compute::malloc<float>(9, context);

    boost::compute::copy(a, a + 9, A, queue);
    boost::compute::copy(b, b + 9, B, queue);
    boost::compute::copy(c, c + 9, C, queue);

    // C = A * A
    boost::compute::blas::gemm(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        boost::compute::blas::no_transpose,
        3, 3, 3,
        1.0f,
        A, 3,
        A, 3,
        0.0f,
        C, 3,
        queue
    );

    boost::compute::copy(C, C + 9, c, queue);
    BOOST_CHECK_CLOSE(c[0], 30.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[1], 36.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[2], 42.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[3], 66.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[4], 81.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[5], 96.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[6], 102.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[7], 126.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[8], 150.0f, 1e-4);

    // C = A * B
    boost::compute::blas::gemm(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        boost::compute::blas::no_transpose,
        3, 3, 3,
        1.0f,
        A, 3,
        B, 3,
        0.0f,
        C, 3,
        queue
    );

    boost::compute::copy(C, C + 9, c, queue);
    BOOST_CHECK_CLOSE(c[0], 169.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[1], 176.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[2], 183.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[3], 382.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[4], 401.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[5], 420.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[6], 595.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[7], 626.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[8], 657.0f, 1e-4);
}

BOOST_AUTO_TEST_CASE(gemm_float2x3)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    float a[6] = { 2.0f, 1.0f,
                   4.0f, 3.0f,
                   6.0f, 5.0f };
    float b[6] = { 1.0f, 2.0f, 3.0f,
                   4.0f, 5.0f, 6.0f };
    float c[9] = { 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f };

    boost::compute::device_ptr<float> A =
        boost::compute::malloc<float>(6, context);
    boost::compute::device_ptr<float> B =
        boost::compute::malloc<float>(6, context);
    boost::compute::device_ptr<float> C =
        boost::compute::malloc<float>(9, context);

    boost::compute::copy(a, a + 6, A, queue);
    boost::compute::copy(b, b + 6, B, queue);
    boost::compute::copy(c, c + 9, C, queue);

    // C = A * B
    boost::compute::blas::gemm(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        boost::compute::blas::no_transpose,
        3, 3, 2,
        1.0f,
        A, 2,
        B, 3,
        0.0f,
        C, 3,
        queue
    );

    boost::compute::copy(C, C + 9, c, queue);
    BOOST_CHECK_CLOSE(c[0], 6.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[1], 9.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[2], 12.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[3], 16.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[4], 23.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[5], 30.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[6], 26.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[7], 37.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[8], 48.0f, 1e-4);

    // C = B * A
    boost::compute::blas::gemm(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        boost::compute::blas::no_transpose,
        2, 2, 3,
        1.0f,
        B, 3,
        A, 2,
        0.0f,
        C, 2,
        queue
    );

    boost::compute::copy(C, C + 4, c, queue);
    BOOST_CHECK_CLOSE(c[0], 28.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[1], 22.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[2], 64.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[3], 49.0f, 1e-4);

    // C = B * A (with alpha = 2)
    boost::compute::blas::gemm(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        boost::compute::blas::no_transpose,
        2, 2, 3,
        2.0f,
        B, 3,
        A, 2,
        0.0f,
        C, 2,
        queue
    );

    boost::compute::copy(C, C + 4, c, queue);
    BOOST_CHECK_CLOSE(c[0], 56.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[1], 44.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[2], 128.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[3], 98.0f, 1e-4);

    // fill C with 4's
    boost::compute::fill(C, C + 4, 4.0f, queue);

    // C = B * A (with beta = 3)
    boost::compute::blas::gemm(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        boost::compute::blas::no_transpose,
        2, 2, 3,
        1.0f,
        B, 3,
        A, 2,
        3.0f,
        C, 2,
        queue
    );

    boost::compute::copy(C, C + 4, c, queue);
    BOOST_CHECK_CLOSE(c[0], 40.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[1], 34.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[2], 76.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[3], 61.0f, 1e-4);

    // fill C with 3's
    boost::compute::fill(C, C + 4, 3.0f, queue);

    // C = B * A (with alpha = 3, beta = 2)
    boost::compute::blas::gemm(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        boost::compute::blas::no_transpose,
        2, 2, 3,
        3.0f,
        B, 3,
        A, 2,
        2.0f,
        C, 2,
        queue
    );

    boost::compute::copy(C, C + 4, c, queue);
    BOOST_CHECK_CLOSE(c[0], 90.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[1], 72.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[2], 198.0f, 1e-4);
    BOOST_CHECK_CLOSE(c[3], 153.0f, 1e-4);
}
