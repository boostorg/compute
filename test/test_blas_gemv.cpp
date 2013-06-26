//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestBlasGemv
#include <boost/test/unit_test.hpp>

#include <boost/compute/malloc.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/blas/gemv.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(gemv_float)
{
    float matrix[9] = { 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f };
    boost::compute::device_ptr<float> A =
        boost::compute::malloc<float>(9, context);
    boost::compute::copy(matrix, matrix + 9, A, queue);

    float input_vector[3] = { 1.0f, 2.0f, 3.0f };
    boost::compute::device_ptr<float> X =
        boost::compute::malloc<float>(3, context);
    boost::compute::copy(input_vector, input_vector + 3, X, queue);

    boost::compute::device_ptr<float> Y =
        boost::compute::malloc<float>(3, context);

    boost::compute::blas::gemv(
        boost::compute::blas::row_major,
        boost::compute::blas::no_transpose,
        3, 3,
        1.0f,
        A, 3,
        X, 1,
        0.0f,
        Y, 1,
        queue
    );

    float output_vector[3];
    boost::compute::copy(Y, Y + 3, output_vector, queue);
    BOOST_CHECK_CLOSE(output_vector[0], 14.0f, 1e-3f);
    BOOST_CHECK_CLOSE(output_vector[1], 32.0f, 1e-3f);
    BOOST_CHECK_CLOSE(output_vector[2], 50.0f, 1e-3f);
}

BOOST_AUTO_TEST_SUITE_END()
