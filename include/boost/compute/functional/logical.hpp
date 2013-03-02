//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_FUNCTIONAL_LOGICAL_HPP
#define BOOST_COMPUTE_FUNCTIONAL_LOGICAL_HPP

namespace boost {
namespace compute {
namespace detail {

template<class Predicate, class Expr>
class invoked_unary_negate_function
{
public:
    typedef int result_type;

    invoked_unary_negate_function(const Predicate &pred,
                                  const Expr &expr)
        : m_pred(pred),
          m_expr(expr)
    {
    }

    Predicate pred() const
    {
        return m_pred;
    }

    Expr expr() const
    {
        return m_expr;
    }

private:
    Predicate m_pred;
    Expr m_expr;
};

template<class Predicate, class Expr1, class Expr2>
class invoked_binary_negate_function
{
public:
    typedef int result_type;

    invoked_binary_negate_function(const Predicate &pred,
                                   const Expr1 &expr1,
                                   const Expr2 &expr2)
        : m_pred(pred),
          m_expr1(expr1),
          m_expr2(expr2)
    {
    }

    Predicate pred() const
    {
        return m_pred;
    }

    Expr1 expr1() const
    {
        return m_expr1;
    }

    Expr2 expr2() const
    {
        return m_expr2;
    }

private:
    Predicate m_pred;
    Expr1 m_expr1;
    Expr2 m_expr2;
};

} // end detail namespace

template<class Arg, class Result>
struct unary_function
{
    typedef Arg argument_type;
    typedef Result result_type;
};

template<class Arg1, class Arg2, class Result>
struct binary_function
{
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Result result_type;
};

template<class Arg1, class Arg2, class Arg3, class Result>
struct ternary_function
{
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Arg3 third_argument_type;
    typedef Result result_type;
};

template<class Predicate>
class unary_negate : public unary_function<void, int>
{
public:
    explicit unary_negate(Predicate pred)
        : m_pred(pred)
    {
    }

    template<class Arg>
    detail::invoked_unary_negate_function<Predicate, Arg>
    operator()(const Arg &arg) const
    {
        return detail::invoked_unary_negate_function<
                   Predicate,
                   Arg
                >(m_pred, arg);
    }

private:
    Predicate m_pred;
};

template<class Predicate>
class binary_negate : public binary_function<void, void, int>
{
public:
    explicit binary_negate(Predicate pred)
        : m_pred(pred)
    {
    }

    template<class Arg1, class Arg2>
    detail::invoked_binary_negate_function<Predicate, Arg1, Arg2>
    operator()(const Arg1 &arg1, const Arg2 &arg2) const
    {
        return detail::invoked_binary_negate_function<
                   Predicate,
                   Arg1,
                   Arg2
                >(m_pred, arg1, arg2);
    }

private:
    Predicate m_pred;
};

template<class T>
struct logical_not : public unary_function<T, int>
{
    template<class Expr>
    detail::invoked_unary_function<Expr, int>
    operator()(const Expr &expr) const
    {
        return detail::invoked_unary_function<Expr, int>("!", expr);
    }
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_FUNCTIONAL_LOGICAL_HPP
