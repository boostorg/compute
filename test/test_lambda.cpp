//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestLambda
#include <boost/test/unit_test.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(squared_plus_one)
{
    bc::vector<int> vector;
    vector.push_back(1);
    vector.push_back(2);
    vector.push_back(3);
    vector.push_back(4);
    vector.push_back(5);

    // multiply each value by itself and add one
    bc::transform(vector.begin(),
                  vector.end(),
                  vector.begin(),
                  (bc::_1 * bc::_1) + 1);

    BOOST_CHECK_EQUAL(vector[0], 2);
    BOOST_CHECK_EQUAL(vector[1], 5);
    BOOST_CHECK_EQUAL(vector[2], 10);
    BOOST_CHECK_EQUAL(vector[3], 17);
    BOOST_CHECK_EQUAL(vector[4], 26);
}

BOOST_AUTO_TEST_CASE(abs_int)
{
    bc::vector<int> vector;
    vector.push_back(-1);
    vector.push_back(-2);
    vector.push_back(3);
    vector.push_back(-4);
    vector.push_back(5);

    bc::transform(vector.begin(),
                  vector.end(),
                  vector.begin(),
                  abs(bc::_1));

    BOOST_CHECK_EQUAL(vector[0], 1);
    BOOST_CHECK_EQUAL(vector[1], 2);
    BOOST_CHECK_EQUAL(vector[2], 3);
    BOOST_CHECK_EQUAL(vector[3], 4);
    BOOST_CHECK_EQUAL(vector[4], 5);
}

template<class Result, class Expr>
void check_lambda_result(const Expr &)
{
    BOOST_STATIC_ASSERT((
        boost::is_same<
            typename ::boost::compute::lambda::result_of<Expr>::type,
            Result
        >::value
    ));
}

template<class Result, class Expr, class Arg1>
void check_unary_lambda_result(const Expr &, const Arg1 &)
{
    BOOST_STATIC_ASSERT((
        boost::is_same<
            typename ::boost::compute::lambda::result_of<
                Expr,
                typename boost::mpl::vector<Arg1>
            >::type,
            Result
        >::value
    ));
}

template<class Result, class Expr, class Arg1, class Arg2>
void check_binary_lambda_result(const Expr &, const Arg1 &, const Arg2 &)
{
    BOOST_STATIC_ASSERT((
        boost::is_same<
            typename ::boost::compute::lambda::result_of<
                Expr,
                typename boost::mpl::vector<Arg1, Arg2>
            >::type,
            Result
        >::value
    ));
}

BOOST_AUTO_TEST_CASE(result_of)
{
    using ::boost::compute::lambda::_1;
    using ::boost::compute::lambda::_2;

    namespace proto = ::boost::proto;

    check_lambda_result<int>(proto::lit(1));
    check_lambda_result<int>(proto::lit(1) + 2);
    check_lambda_result<float>(proto::lit(1.2f));
    check_lambda_result<float>(proto::lit(1) + 1.2f);
    check_lambda_result<float>(proto::lit(1) / 2 + 1.2f);

    check_unary_lambda_result<int>(_1, int(1));
    check_unary_lambda_result<float>(_1, float(1.2f));
    check_unary_lambda_result<bc::float4_>(_1, bc::float4_(1, 2, 3, 4));
    check_unary_lambda_result<bc::float4_>(2.0f * _1, bc::float4_(1, 2, 3, 4));
    check_unary_lambda_result<bc::float4_>(_1 * 2.0f, bc::float4_(1, 2, 3, 4));

    check_binary_lambda_result<float>(dot(_1, _2), bc::float4_(0, 1, 2, 3), bc::float4_(3, 2, 1, 0));
    check_unary_lambda_result<float>(dot(_1, bc::float4_(3, 2, 1, 0)), bc::float4_(0, 1, 2, 3));

    check_unary_lambda_result<int>(_1 + 2, int(2));
    check_unary_lambda_result<float>(_1 + 2, float(2.2f));

    check_binary_lambda_result<int>(_1 + _2, int(1), int(2));
    check_binary_lambda_result<float>(_1 + _2, int(1), float(2.2f));

    check_unary_lambda_result<int>(_1 + _1, int(1));
    check_unary_lambda_result<float>(_1 * _1, float(1));
}

BOOST_AUTO_TEST_CASE(make_function_from_lamdba)
{
    using boost::compute::_1;

    int data[] = { 2, 4, 6, 8, 10 };
    boost::compute::vector<int> vector(data, data + 5);
    BOOST_CHECK_EQUAL(vector.size(), size_t(5));

//    boost::compute::function<int(int)> f =
//        boost::compute::make_function_from_lambda<int(int)>(_1 * 2 + 3);

    boost::compute::transform(vector.begin(),
                              vector.end(),
                              vector.begin(),
                              boost::compute::make_function_from_lambda<int(int)>(_1 * 2 + 3));
    BOOST_CHECK_EQUAL(int(vector[0]), int(7));
    BOOST_CHECK_EQUAL(int(vector[1]), int(11));
    BOOST_CHECK_EQUAL(int(vector[2]), int(15));
    BOOST_CHECK_EQUAL(int(vector[3]), int(19));
    BOOST_CHECK_EQUAL(int(vector[4]), int(23));
}
