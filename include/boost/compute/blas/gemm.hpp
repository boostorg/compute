//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_BLAS_GEMM_HPP
#define BOOST_COMPUTE_BLAS_GEMM_HPP

#include <boost/compute/cl.hpp>
#include <boost/compute/device_ptr.hpp>
#include <boost/compute/blas/enums.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>

namespace boost {
namespace compute {
namespace blas {

template<class Scalar>
inline void gemm(const matrix_order order,
                 const matrix_transpose trans_a,
                 const matrix_transpose trans_b,
                 const int M,
                 const int N,
                 const int K,
                 const Scalar alpha,
                 device_ptr<Scalar> A,
                 const int lda,
                 device_ptr<Scalar> B,
                 const int ldb,
                 const Scalar beta,
                 device_ptr<Scalar> C,
                 const int ldc,
                 command_queue &queue)
{
    (void) order;
    (void) trans_a;
    (void) trans_b;

    ::boost::compute::detail::meta_kernel k("gemm");
    k.add_arg<Scalar>("alpha", alpha);
    k.add_arg<Scalar>("beta", beta);
    k.add_arg<const cl_uint>("M", static_cast<const cl_uint>(M));
    k.add_arg<const cl_uint>("N", static_cast<const cl_uint>(N));
    k.add_arg<const cl_uint>("K", static_cast<const cl_uint>(K));
    k.add_arg<const cl_uint>("lda", static_cast<const cl_uint>(lda));
    k.add_arg<const cl_uint>("ldb", static_cast<const cl_uint>(ldb));
    k.add_arg<const cl_uint>("ldc", static_cast<const cl_uint>(ldc));
    size_t a_index = k.add_arg<const Scalar *>("__global", "A");
    size_t b_index = k.add_arg<const Scalar *>("__global", "B");
    size_t c_index = k.add_arg<Scalar *>("__global", "C");

    k <<
        k.decl<cl_uint>("i") << " = get_global_id(0);\n" <<
        k.decl<cl_uint>("j") << " = get_global_id(1);\n" <<
        k.decl<Scalar>("sum") << " = 0;\n" <<
        "for(uint k = 0; k < K; k++){\n" <<
        "    sum += " << A[k.expr<cl_uint>("i*lda+k")] << " * "
                      << B[k.expr<cl_uint>("k*ldb+j")] << ";\n" <<
        "};\n" <<
        C[k.expr<cl_uint>("i*ldc+j")] << "=" <<
            "alpha * sum + beta *" << C[k.expr<cl_uint>("i*ldc+j")] << ";\n";

    const context &context = queue.get_context();
    ::boost::compute::kernel kernel = k.compile(context);

    kernel.set_arg(a_index, A.get_buffer());
    kernel.set_arg(b_index, B.get_buffer());
    kernel.set_arg(c_index, C.get_buffer());

    size_t work_group_offsets[] = { 0, 0 };
    size_t work_group_sizes[] = { static_cast<size_t>(N),
                                  static_cast<size_t>(M) };
    queue.enqueue_nd_range_kernel(kernel,
                                  2,
                                  work_group_offsets,
                                  work_group_sizes,
                                  0);
}

} // end blas namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_BLAS_GEMM_HPP
