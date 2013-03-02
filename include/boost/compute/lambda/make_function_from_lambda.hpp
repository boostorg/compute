//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_LAMBDA_MAKE_FUNCTION_FROM_LAMBDA_HPP
#define BOOST_COMPUTE_LAMBDA_MAKE_FUNCTION_FROM_LAMBDA_HPP

#include <boost/compute/function.hpp>
#include <boost/compute/lambda/context.hpp>

namespace boost {
namespace compute {

template<class Signature, class Expr>
class lambda_function : public function<Signature>
{
public:
    template<class Signature_>
    struct result
    {
        typedef typename Expr::template result<Signature_>::type type;
    };

    lambda_function(const Expr &expr)
        : function<Signature>("lambda"),
          m_expr(expr)
    {
    }

    template<class Arg>
    ::boost::compute::lambda::detail::invoked_unary_expression<Expr, Arg>
    operator()(const Arg &arg) const
    {
        return m_expr(arg);
    }

private:
    Expr m_expr;
};

template<class Signature, class Expr>
inline lambda_function<Signature, Expr> make_function_from_lambda(const Expr &expr)
{
    return lambda_function<Signature, Expr>(expr);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_LAMBDA_MAKE_FUNCTION_FROM_LAMBDA_HPP
