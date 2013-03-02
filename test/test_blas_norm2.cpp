//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestBlasNorm2
#include <boost/test/unit_test.hpp>

#include <boost/compute/malloc.hpp>
#include <boost/compute/blas/norm2.hpp>
#include <boost/compute/container/vector.hpp>

BOOST_AUTO_TEST_CASE(norm2_float)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    float data[] = { 1.0f, 2.0f, 4.0f, 8.0f, 16.0f };
    boost::compute::device_ptr<float> X =
        boost::compute::malloc<float>(5, context);
    boost::compute::copy(data, data + 5, X, queue);

    float norm = boost::compute::blas::norm2(5, X, 1, queue);
    BOOST_CHECK_CLOSE(norm, 18.466185312619388f, 1e-4);

    boost::compute::vector<float> vector(data, data + 5, context);
    norm = boost::compute::blas::norm2(5, &vector[0], 1, queue);
    BOOST_CHECK_CLOSE(norm, 18.466185312619388f, 1e-4);
}
