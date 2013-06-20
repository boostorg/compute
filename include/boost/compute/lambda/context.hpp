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

#include <boost/proto/core.hpp>
#include <boost/proto/context.hpp>
#include <boost/type_traits.hpp>

#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/lambda/result_of.hpp>
#include <boost/compute/lambda/functional.hpp>
#include <boost/compute/detail/meta_kernel.hpp>

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
template<class Args>
struct context : proto::callable_context<context<Args> >
{
    typedef void result_type;

    context(boost::compute::detail::meta_kernel &kernel,
            const Args &args)
        : stream(kernel),
          m_args(args)
    {
    }

    ~context()
    {
    }

    // handle terminals
    template<class T>
    void operator()(proto::tag::terminal, const T &x)
    {
        // terminal values in lambda expressions are always literals
        stream << stream.lit(x);
    }

    // handle placeholders
    template<int I>
    void operator()(proto::tag::terminal, placeholder<I>)
    {
        stream << boost::get<I>(m_args);
    }

    // handle functions
    template<class F, class Arg>
    void operator()(proto::tag::function,
                    const F &function,
                    const Arg &arg)
    {
        apply_function(proto::value(function), arg);
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

    template<class F, class Arg>
    void apply_function(const F &function, const Arg &arg)
    {
        stream << function.function_name() << '(';
        proto::eval(arg, *this);
        stream << ')';
    }

    template<size_t N, class Arg>
    void apply_function(detail::get_func<N>, const Arg &arg)
    {
        typedef typename
            boost::remove_cv<
                typename boost::compute::lambda::result_of<Arg, Args>::type
            >::type T;

        apply_get_function<N, T>(arg, typename proto::tag_of<Arg>::type());
    }

    template<size_t N, class T, class Arg, class Tag>
    void apply_get_function(const Arg &arg, Tag)
    {
        proto::eval(arg, *this);
        stream << detail::get_func_suffix<N, T>::value();
    }

    template<size_t N, class T, class Arg>
    void apply_get_function(const Arg &arg, proto::tag::terminal)
    {
        apply_get_terminal_function<N, T>(arg, proto::value(arg));
    }

    template<size_t N, class T, class Arg, class ArgValue>
    void apply_get_terminal_function(const Arg &arg, ArgValue)
    {
        proto::eval(arg, *this);
        stream << detail::get_func_suffix<N, T>::value();
    }

    template<size_t N, class T, class Arg, int I>
    void apply_get_terminal_function(Arg, placeholder<I>)
    {
        stream << ::boost::compute::get<N>()(::boost::get<I>(m_args));
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
    BOOST_COMPUTE_LAMBDA_CONTEXT_DEFINE_BINARY_OPERATOR(proto::tag::assign, '=')

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

    boost::compute::detail::meta_kernel &stream;
    Args m_args;
};

namespace detail {

template<class Expr, class Arg>
struct invoked_unary_expression
{
    typedef typename ::boost::tr1_result_of<Expr(Arg)>::type result_type;

    invoked_unary_expression(const Expr &expr, const Arg &arg)
        : m_expr(expr),
          m_arg(arg)
    {
    }

    Expr m_expr;
    Arg m_arg;
};

template<class Expr, class Arg>
boost::compute::detail::meta_kernel&
operator<<(boost::compute::detail::meta_kernel &kernel,
           const invoked_unary_expression<Expr, Arg> &expr)
{
    context<boost::tuple<Arg> > ctx(kernel, boost::make_tuple(expr.m_arg));
    proto::eval(expr.m_expr, ctx);

    return kernel;
}

template<class Expr, class Arg1, class Arg2>
struct invoked_binary_expression
{
    typedef typename ::boost::tr1_result_of<Expr(Arg1, Arg2)>::type result_type;

    invoked_binary_expression(const Expr &expr,
                              const Arg1 &arg1,
                              const Arg2 &arg2)
        : m_expr(expr),
          m_arg1(arg1),
          m_arg2(arg2)
    {
    }

    Expr m_expr;
    Arg1 m_arg1;
    Arg2 m_arg2;
};

template<class Expr, class Arg1, class Arg2>
boost::compute::detail::meta_kernel&
operator<<(boost::compute::detail::meta_kernel &kernel,
           const invoked_binary_expression<Expr, Arg1, Arg2> &expr)
{
    context<boost::tuple<Arg1, Arg2> > ctx(
        kernel,
        boost::make_tuple(expr.m_arg1, expr.m_arg2)
    );
    proto::eval(expr.m_expr, ctx);

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
                typename boost::tuple<Arg>
            >::type type;
    };

    template<class This, class Arg1, class Arg2>
    struct result<This(Arg1, Arg2)>
    {
        typedef typename
            ::boost::compute::lambda::result_of<
                Expr,
                typename boost::tuple<Arg1, Arg2>
            >::type type;
    };

    template<class Arg>
    detail::invoked_unary_expression<expression<Expr>, Arg>
    operator()(const Arg &x) const
    {
        return detail::invoked_unary_expression<expression<Expr>, Arg>(*this, x);
    }

    template<class Arg1, class Arg2>
    detail::invoked_binary_expression<expression<Expr>, Arg1, Arg2>
    operator()(const Arg1 &x, const Arg2 &y) const
    {
        return detail::invoked_binary_expression<
                   expression<Expr>,
                   Arg1,
                   Arg2
                >(*this, x, y);
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
