//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_BLAS_YAX_HPP
#define BOOST_COMPUTE_BLAS_YAX_HPP

#include <boost/compute/types.hpp>
#include <boost/compute/lambda.hpp>
#include <boost/compute/device_ptr.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/type_traits/make_vector_type.hpp>

namespace boost {
namespace compute {
namespace blas {

template<class Scalar>
inline void yax(const int N,
                Scalar alpha,
                device_ptr<Scalar> X,
                const int incX,
                device_ptr<Scalar> Y,
                const int incY,
                command_queue &queue)
{
    BOOST_ASSERT_MSG(incX == 1, "Only contiguous arrays are currently supported.");
    BOOST_ASSERT_MSG(incY == 1, "Only contiguous arrays are currently supported.");

    using ::boost::compute::lambda::_1;

    if(N % 8 == 0){
        typedef typename make_vector_type<Scalar, 8>::type Scalar8;

        const int N8 = N / 8;
        device_ptr<Scalar8> X8 = X.template cast<Scalar8>();
        device_ptr<Scalar8> Y8 = Y.template cast<Scalar8>();

        ::boost::compute::transform(X8, X8 + N8, Y8, alpha * _1, queue);
    }
    else if(N % 4 == 0){
        typedef typename make_vector_type<Scalar, 4>::type Scalar4;

        const int N4 = N / 4;
        device_ptr<Scalar4> X4 = X.template cast<Scalar4>();
        device_ptr<Scalar4> Y4 = Y.template cast<Scalar4>();

        ::boost::compute::transform(X4, X4 + N4, Y4, alpha * _1, queue);
    }
    else if(N % 2 == 0){
        typedef typename make_vector_type<Scalar, 2>::type Scalar2;

        const int N2 = N / 2;
        device_ptr<Scalar2> X2 = X.template cast<Scalar2>();
        device_ptr<Scalar2> Y2 = Y.template cast<Scalar2>();

        ::boost::compute::transform(X2, X2 + N2, Y2, alpha * _1, queue);
    }
    else {
        ::boost::compute::transform(X, X + N, Y, alpha * _1, queue);
    }
}

} // end blas namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_BLAS_YAX_HPP
