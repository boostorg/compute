//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestBlasAxpy
#include <boost/test/unit_test.hpp>

#include <boost/compute/malloc.hpp>
#include <boost/compute/blas/axpy.hpp>

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(saxpy)
{
    // setup context and queue for default device
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    // create input vector X
    float x_data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    compute::device_ptr<float> x = compute::malloc<float>(8, context);
    compute::copy(x_data, x_data + 8, x, queue);

    // create input vector Y
    float y_data[] = { 1.f, 0.f, 2.f, 0.f, 3.f, 0.f, 4.f, 0.f };
    compute::device_ptr<float> y = compute::malloc<float>(8, context);
    compute::copy(y_data, y_data + 8, y, queue);

    // run saxpy
    compute::blas::axpy(
        8, // n
        3.f, // alpha
        x,
        1, // incX
        y,
        1, // incY
        queue
    );

    // copy results to host
    float result[8];
    compute::copy(y, y + 8, result, queue);
    queue.finish();

    // check result values
    BOOST_CHECK_CLOSE(result[0], 4.f, 1e-4);
    BOOST_CHECK_CLOSE(result[1], 6.f, 1e-4);
    BOOST_CHECK_CLOSE(result[2], 11.f, 1e-4);
    BOOST_CHECK_CLOSE(result[3], 12.f, 1e-4);
    BOOST_CHECK_CLOSE(result[4], 18.f, 1e-4);
    BOOST_CHECK_CLOSE(result[5], 18.f, 1e-4);
    BOOST_CHECK_CLOSE(result[6], 25.f, 1e-4);
    BOOST_CHECK_CLOSE(result[7], 24.f, 1e-4);

    // free device memory
    compute::free(x);
    compute::free(y);
}
