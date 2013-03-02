//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestBlasIamax
#include <boost/test/unit_test.hpp>

#include <boost/compute/malloc.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/blas/iamax.hpp>

BOOST_AUTO_TEST_CASE(iamax_float)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    float data[] = { 1.0f, 3.0f, 2.0f, 4.0f, 2.0f };
    boost::compute::device_ptr<float> X =
        boost::compute::malloc<float>(5, context);
    boost::compute::copy(data, data + 5, X, queue);
    BOOST_CHECK_EQUAL(boost::compute::blas::iamax(5, X, 1, queue), 3);
}
