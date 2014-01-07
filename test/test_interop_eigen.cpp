//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestInteropEigen
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/interop/eigen.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bcl = boost::compute;

BOOST_AUTO_TEST_CASE(eigen)
{
    Eigen::MatrixXf mat(3, 3);
    mat << 1, 2, 3,
           6, 5, 4,
           7, 8, 9;

    // copy matrix to gpu buffer
    bcl::vector<float> vec(9, context);
    bcl::eigen_copy_matrix_to_buffer(mat, vec.begin(), queue);
    CHECK_RANGE_EQUAL(float, 9, vec, (1, 6, 7, 2, 5, 8, 3, 4, 9));

    // transpose matrix and then copy to gpu buffer
    mat = mat.transpose().eval();
    bcl::eigen_copy_matrix_to_buffer(mat, vec.begin(), queue);
    CHECK_RANGE_EQUAL(float, 9, vec, (1, 2, 3, 6, 5, 4, 7, 8, 9));

    // set matrix to zero and copy data back from gpu buffer
    mat.setZero();
    bcl::eigen_copy_buffer_to_matrix(vec.begin(), mat, queue);
    BOOST_CHECK(mat.isZero() == false);
    BOOST_CHECK_EQUAL(mat.sum(), 45);
}

BOOST_AUTO_TEST_CASE(eigen_types)
{
    BOOST_CHECK(std::strcmp(bcl::type_name<Eigen::Vector2i>(), "int2") == 0);
    BOOST_CHECK(std::strcmp(bcl::type_name<Eigen::Vector2f>(), "float2") == 0);
    BOOST_CHECK(std::strcmp(bcl::type_name<Eigen::Vector4f>(), "float4") == 0);
    BOOST_CHECK(std::strcmp(bcl::type_name<Eigen::Vector4d>(), "double4") == 0);
}

BOOST_AUTO_TEST_CASE(multiply_matrix4)
{
    std::vector<Eigen::Vector4f> host_vectors;
    std::vector<Eigen::Matrix4f> host_matrices;

    host_matrices.push_back(Eigen::Matrix4f::Identity());
    host_matrices.push_back(Eigen::Matrix4f::Zero());
    host_matrices.push_back(Eigen::Matrix4f::Ones());
    host_matrices.push_back(Eigen::Matrix4f::Ones() * 2);

    host_vectors.push_back(Eigen::Vector4f(1, 2, 3, 4));
    host_vectors.push_back(Eigen::Vector4f(4, 3, 2, 1));
    host_vectors.push_back(Eigen::Vector4f(1, 2, 3, 4));
    host_vectors.push_back(Eigen::Vector4f(4, 3, 2, 1));

    // returns the result of M*x
    BOOST_COMPUTE_FUNCTION(Eigen::Vector4f, transform4x4, (const Eigen::Matrix4f, const Eigen::Vector4f),
    {
        float4 r;
        r.x = dot(_1.s048c, _2);
        r.y = dot(_1.s159d, _2);
        r.z = dot(_1.s26ae, _2);
        r.w = dot(_1.s37bf, _2);
        return r;
    });

    bcl::vector<Eigen::Vector4f> vectors(4, context);
    bcl::vector<Eigen::Matrix4f> matrices(4, context);
    bcl::vector<Eigen::Vector4f> results(4, context);

    bcl::copy(host_vectors.begin(), host_vectors.end(), vectors.begin(), queue);
    bcl::copy(host_matrices.begin(), host_matrices.end(), matrices.begin(), queue);

    bcl::transform(
        matrices.begin(), matrices.end(), vectors.begin(), results.begin(), transform4x4, queue
    );

    std::vector<Eigen::Vector4f> host_results(4);
    bcl::copy(results.begin(), results.end(), host_results.begin(), queue);

    BOOST_CHECK((host_matrices[0] * host_vectors[0]) == host_results[0]);
    BOOST_CHECK((host_matrices[1] * host_vectors[1]) == host_results[1]);
    BOOST_CHECK((host_matrices[2] * host_vectors[2]) == host_results[2]);
    BOOST_CHECK((host_matrices[3] * host_vectors[3]) == host_results[3]);
}

BOOST_AUTO_TEST_SUITE_END()
