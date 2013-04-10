//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_LAMBDA_CONTEXT_HPP
#define BOOST_COMPUTE_LAMBDA_CONTEXT_HPP

#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/proto/core.hpp>
#include <boost/proto/context.hpp>
#include <boost/type_traits.hpp>

#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/lambda/result_of.hpp>

namespace boost {
namespace compute {
namespace lambda {

namespace mpl = boost::mpl;
namespace proto = boost::proto;

// placeholder type
template<int I>
struct placeholder
{
};

#define BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATOR(scalar, n) \
    void operator()(proto::tag::terminal, \
                    const BOOST_COMPUTE_MAKE_VECTOR_TYPE(scalar, n) &x) \
    { \
        typedef BOOST_COMPUTE_MAKE_VECTOR_TYPE(scalar, n) type; \
        stream << '(' << type_name<type>() << ")("; \
        for(size_t i = 0; i < n; i++){ \
            stream << x[i]; \
            if(i != n - 1){ \
                stream << ','; \
            } \
        } \
        stream << ')'; \
    }

#define BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(scalar) \
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATOR(scalar, 2) \
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATOR(scalar, 4) \
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATOR(scalar, 8) \
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATOR(scalar, 16)

#define BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(tag, op) \
    template<class LHS, class RHS> \
    void operator()(tag, const LHS &lhs, const RHS &rhs) \
    { \
        if(proto::arity_of<LHS>::value > 0){ \
            stream << '('; \
            proto::eval(lhs, *this); \
            stream << ')'; \
        } \
        else { \
            proto::eval(lhs, *this); \
        } \
        \
        stream << op; \
        \
        if(proto::arity_of<RHS>::value > 0){ \
            stream << '('; \
            proto::eval(rhs, *this); \
            stream << ')'; \
        } \
        else { \
            proto::eval(rhs, *this); \
        } \
    }

// lambda expression context
struct context : proto::callable_context<context>
{
    typedef void result_type;

    ~context()
    {
    }

    // handle terminals
    template<class T>
    void operator()(proto::tag::terminal, const T &x)
    {
        stream << x;
    }

    void operator()(proto::tag::terminal, const float &x)
    {
        stream << std::showpoint << x << "f";
    }

    // handle vector terminals
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(char)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(uchar)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(short)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(ushort)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(int)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(uint)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(long)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(ulong)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(float)
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_VECTOR_TERMINAL_OPERATORS(double)

    // handle placeholders
    template<int I>
    void operator()(proto::tag::terminal, placeholder<I>)
    {
        stream << "$" << I + 1;
    }

    // handle functions
    template<class F, class Args>
    void operator()(proto::tag::function,
                    const F &function,
                    const Args &args)
    {
        stream << proto::value(function).function_name() << '(';
        proto::eval(args, *this);
        stream << ')';
    }

    template<class F, class Arg1, class Arg2>
    void operator()(proto::tag::function,
                    const F &function,
                    const Arg1 &arg1,
                    const Arg2 &arg2)
    {
        stream << proto::value(function).function_name() << '(';
        proto::eval(arg1, *this);
        stream << ',';
        proto::eval(arg2, *this);
        stream << ')';
    }

    template<class F, class Arg1, class Arg2, class Arg3>
    void operator()(proto::tag::function,
                    const F &function,
                    const Arg1 &arg1,
                    const Arg2 &arg2,
                    const Arg3 &arg3)
    {
        stream << proto::value(function).function_name() << '(';
        proto::eval(arg1, *this);
        stream << ',';
        proto::eval(arg2, *this);
        stream << ',';
        proto::eval(arg3, *this);
        stream << ')';
    }

    // operators
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::plus, '+')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::minus, '-')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::multiplies, '*')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::divides, '/')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::modulus, '%')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::less, '<')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::greater, '>')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::less_equal, "<=")
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::greater_equal, ">=")
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::equal_to, "==")
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::not_equal_to, "!=")
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::logical_and, "&&")
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::logical_or, "||")
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::bitwise_and, '&')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::bitwise_or, '|')
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::bitwise_xor, '^')

    // subscript operator
    template<class LHS, class RHS>
    void operator()(proto::tag::subscript, const LHS &lhs, const RHS &rhs)
    {
        proto::eval(lhs, *this);
        stream << '[';
        proto::eval(rhs, *this);
        stream << ']';
    }

    // ternary conditional operator
    template<class Pred, class Arg1, class Arg2>
    void operator()(proto::tag::if_else_, const Pred &p, const Arg1 &x, const Arg2 &y)
    {
        proto::eval(p, *this);
        stream << '?';
        proto::eval(x);
        stream << ':';
        proto::eval(y);
    }

    std::stringstream stream;
};

namespace detail {

template<class Expr, class Arg>
struct invoked_unary_expression
{
    typedef typename ::boost::tr1_result_of<Expr(Arg)>::type result_type;

    invoked_unary_expression(const std::string &expr, const Arg &arg)
        : m_expr(expr),
          m_arg(arg)
    {
    }

    std::string m_expr;
    Arg m_arg;
};

template<class Expr, class Arg>
boost::compute::detail::meta_kernel&
operator<<(boost::compute::detail::meta_kernel &kernel,
           const invoked_unary_expression<Expr, Arg> &expr)
{
    std::string s = expr.m_expr;
    const size_t token_length = 2;

    size_t i = 0;
    while(true){
        size_t a1 = s.find("$1", i);

        if(a1 == std::string::npos){
            // reached end of string
            kernel << s.substr(i, a1);
            break;
        }
        else {
            kernel << s.substr(i, (a1 - i));
            kernel << expr.m_arg;
            i = a1 + token_length;
        }
    }

    return kernel;
}

template<class Expr, class Arg1, class Arg2>
struct invoked_binary_expression
{
    typedef typename ::boost::tr1_result_of<Expr(Arg1, Arg2)>::type result_type;

    invoked_binary_expression(const std::string &expr,
                              const Arg1 &arg1,
                              const Arg2 &arg2)
        : m_expr(expr),
          m_arg1(arg1),
          m_arg2(arg2)
    {
    }

    std::string m_expr;
    Arg1 m_arg1;
    Arg2 m_arg2;
};

template<class Expr, class Arg1, class Arg2>
boost::compute::detail::meta_kernel&
operator<<(boost::compute::detail::meta_kernel &kernel,
           const invoked_binary_expression<Expr, Arg1, Arg2> &expr)
{
    std::string s = expr.m_expr;
    const size_t token_length = 2;

    size_t i = 0;
    while(true){
        size_t a1 = s.find("$1", i);
        size_t a2 = s.find("$2", i);

        if(a1 == std::string::npos && a2 == std::string::npos){
            // reached end of string
            kernel << s.substr(i, a1);
            break;
        }
        else if(a1 < a2){
            kernel << s.substr(i, (a1 - i));
            kernel << expr.m_arg1;
            i = a1 + token_length;
        }
        else if(a2 < a1){
            kernel << s.substr(i, (a2 - i));
            kernel << expr.m_arg2;
            i = a2 + token_length;
        }
    }

    return kernel;
}

} // end detail namespace

// forward declare domain
struct domain;

// lambda expression wrapper
template<class Expr>
struct expression : proto::extends<Expr, expression<Expr>, domain>
{
    typedef proto::extends<Expr, expression<Expr>, domain> base_type;

    BOOST_PROTO_EXTENDS_USING_ASSIGN(expression)

    expression(const Expr &expr = Expr())
        : base_type(expr)
    {
    }

    // result_of protocol
    template<class Signature>
    struct result
    {
    };

    template<class This>
    struct result<This()>
    {
        typedef
            typename ::boost::compute::lambda::result_of<Expr>::type type;
    };

    template<class This, class Arg>
    struct result<This(Arg)>
    {
        typedef
            typename ::boost::compute::lambda::result_of<
                Expr,
                typename mpl::vector<Arg>
            >::type type;
    };

    template<class This, class Arg1, class Arg2>
    struct result<This(Arg1, Arg2)>
    {
        typedef typename
            ::boost::compute::lambda::result_of<
                Expr,
                typename mpl::vector<Arg1, Arg2>
            >::type type;
    };

    ::boost::compute::detail::meta_kernel_variable<
        typename ::boost::tr1_result_of<expression<Expr>()>::type
    >
    operator()() const
    {
        typedef typename
            ::boost::tr1_result_of<expression<Expr>()>::type
            result_type;

        context ctx;
        proto::eval(*this, ctx);

        return ::boost::compute::detail::meta_kernel::make_expr<result_type>(ctx.stream.str());
    }

    template<class Arg>
    detail::invoked_unary_expression<expression<Expr>, Arg>
    operator()(const Arg &x) const
    {
        context ctx;
        proto::eval(*this, ctx);
        std::string expr = ctx.stream.str();

        return detail::invoked_unary_expression<expression<Expr>, Arg>(expr, x);
    }

    template<class Arg1, class Arg2>
    detail::invoked_binary_expression<expression<Expr>, Arg1, Arg2>
    operator()(const Arg1 &x, const Arg2 &y) const
    {
        context ctx;
        proto::eval(*this, ctx);
        std::string expr = ctx.stream.str();

        return detail::invoked_binary_expression<expression<Expr>,
                                                 Arg1,
                                                 Arg2>(expr, x, y);
    }
};

// lambda expression domain
struct domain : proto::domain<proto::generator<expression> >
{
};

} // end lambda namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_LAMBDA_CONTEXT_HPP
