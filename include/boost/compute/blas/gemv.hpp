//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_BLAS_GEMV_HPP
#define BOOST_COMPUTE_BLAS_GEMV_HPP

#include <boost/compute/cl.hpp>
#include <boost/compute/device_ptr.hpp>
#include <boost/compute/blas/enums.hpp>
#include <boost/compute/detail/meta_kernel.hpp>

namespace boost {
namespace compute {
namespace blas {

template<class Scalar>
inline void gemv(const matrix_order order,
                 const matrix_transpose trans_a,
                 const int M,
                 const int N,
                 const Scalar alpha,
                 device_ptr<Scalar> A,
                 const int lda,
                 device_ptr<Scalar> X,
                 const int incX,
                 const Scalar beta,
                 device_ptr<Scalar> Y,
                 const int incY,
                 command_queue &queue)
{
    BOOST_ASSERT_MSG(incX == 1, "Only contiguous arrays are currently supported.");
    BOOST_ASSERT_MSG(incY == 1, "Only contiguous arrays are currently supported.");

    (void) order;
    (void) trans_a;
    (void) lda;

    ::boost::compute::detail::meta_kernel kernel("gemv");
    kernel.add_set_arg<Scalar>("alpha", alpha);
    kernel.add_set_arg<Scalar>("beta", beta);
    kernel.add_set_arg<const cl_uint>("M", static_cast<const cl_uint>(M));
    kernel.add_set_arg<const cl_uint>("N", static_cast<const cl_uint>(N));
    size_t a_index = kernel.add_arg<const Scalar *>("__global", "A");
    size_t x_index = kernel.add_arg<const Scalar *>("__global", "X");
    size_t y_index = kernel.add_arg<Scalar *>("__global", "Y");

    ::boost::compute::detail::meta_kernel_variable<cl_uint> i =
        kernel.var<cl_uint>("get_global_id(0)");
    ::boost::compute::detail::meta_kernel_variable<cl_uint> k =
        kernel.var<cl_uint>("k");
    kernel <<
        kernel.decl<cl_uint>("i") << " = get_global_id(0);\n" <<
        kernel.decl<Scalar>("sum") << " = 0;\n" <<
        "for(uint k = 0; k < N; k++){\n" <<
        "    sum += " << A[kernel.expr<cl_uint>("k + M*i")] << " * " << X[k] << ";\n" <<
        "};\n" <<
        Y[i] << " = " << "alpha * sum + beta * " << Y[i] << ";";

    const context &context = queue.get_context();
    ::boost::compute::kernel _kernel = kernel.compile(context);

    _kernel.set_arg(a_index, A.get_buffer());
    _kernel.set_arg(x_index, X.get_buffer());
    _kernel.set_arg(y_index, Y.get_buffer());

    queue.enqueue_1d_range_kernel(_kernel, 0, static_cast<std::size_t>(N));
}

} // end blas namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_BLAS_GEMV_HPP
