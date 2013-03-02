//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_FUNCTION_HPP
#define BOOST_COMPUTE_FUNCTION_HPP

#include <string>

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/function_traits.hpp>

#include <boost/compute/cl.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class Result>
class invoked_nullary_function
{
public:
    typedef Result result_type;

    invoked_nullary_function(const std::string &name)
        : m_name(name)
    {
    }

    invoked_nullary_function(const std::string &name,
                             const std::string &source)
        : m_name(name),
          m_source(source)
    {
    }

    std::string name() const
    {
        return m_name;
    }

    std::string source() const
    {
        return m_source;
    }

private:
    std::string m_name;
    std::string m_source;
};

template<class Expr, class Result>
class invoked_unary_function
{
public:
    typedef Result result_type;

    invoked_unary_function(const std::string &name,
                           const Expr &expr)
        : m_name(name),
          m_expr(expr)
    {
    }

    invoked_unary_function(const std::string &name,
                           const Expr &expr,
                           const std::string &source)
        : m_name(name),
          m_expr(expr),
          m_source(source)
    {
    }

    std::string name() const
    {
        return m_name;
    }

    Expr expr() const
    {
        return m_expr;
    }

    std::string source() const
    {
        return m_source;
    }

private:
    std::string m_name;
    Expr m_expr;
    std::string m_source;
};

template<class Expr1, class Expr2, class Result>
class invoked_binary_function
{
public:
    typedef Result result_type;

    invoked_binary_function(const std::string &name,
                            const Expr1 &arg1,
                            const Expr2 &arg2)
        : m_name(name),
          m_expr1(arg1),
          m_expr2(arg2)
    {
    }

    std::string name() const
    {
        return m_name;
    }

    Expr1 arg1() const
    {
        return m_expr1;
    }

    Expr2 arg2() const
    {
        return m_expr2;
    }

private:
    std::string m_name;
    Expr1 m_expr1;
    Expr2 m_expr2;
};

template<class Expr1, class Expr2, class Expr3, class Result>
class invoked_ternary_function
{
public:
    typedef Result result_type;

    invoked_ternary_function(const std::string &name,
                             const Expr1 &arg1,
                             const Expr2 &arg2,
                             const Expr3 &arg3)
        : m_name(name),
          m_expr1(arg1),
          m_expr2(arg2),
          m_expr3(arg3)
    {
    }

    std::string name() const
    {
        return m_name;
    }

    Expr1 arg1() const
    {
        return m_expr1;
    }

    Expr2 arg2() const
    {
        return m_expr2;
    }

    Expr3 arg3() const
    {
        return m_expr3;
    }

private:
    std::string m_name;
    Expr1 m_expr1;
    Expr2 m_expr2;
    Expr3 m_expr3;
};

} // end detail namespace

template<class Signature>
class function
{
public:
    typedef typename
        boost::function_traits<Signature>::result_type
        result_type;

    BOOST_STATIC_CONSTANT(
      size_t,
      arity = boost::function_traits<Signature>::arity
    );

    function(const std::string &name)
        : m_name(name)
    {
    }

    ~function()
    {
    }

    std::string name() const
    {
        return m_name;
    }

    void set_source(const std::string &source)
    {
        m_source = source;
    }

    std::string source() const
    {
        return m_source;
    }

    detail::invoked_nullary_function<result_type>
    operator()() const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 0,
            "Non-nullary function invoked with zero arguments"
        );

        return detail::invoked_nullary_function<result_type>(m_name, m_source);
    }

    template<class Arg1>
    detail::invoked_unary_function<Arg1, result_type>
    operator()(const Arg1 &arg1) const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 1,
            "Non-unary function invoked one argument"
        );

        return detail::invoked_unary_function<Arg1, result_type>(m_name, arg1, m_source);
    }

    template<class Arg1, class Arg2>
    detail::invoked_binary_function<Arg1, Arg2, result_type>
    operator()(const Arg1 &arg1, const Arg2 &arg2) const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 2,
            "Non-binary function invoked with two arguments"
        );

        return detail::invoked_binary_function<Arg1, Arg2, result_type>(m_name, arg1, arg2);
    }

    template<class Arg1, class Arg2, class Arg3>
    detail::invoked_ternary_function<Arg1, Arg2, Arg3, result_type>
    operator()(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3) const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 3,
            "Non-ternary function invoked with two arguments"
        );

        return detail::invoked_ternary_function<Arg1, Arg2, Arg3, result_type>(m_name, arg1, arg2, arg3);
    }

private:
    std::string m_name;
    std::string m_source;
};

template<class Signature>
inline function<Signature> make_function_from_source(const std::string &name,
                                                     const std::string &source)
{
    function<Signature> f(name);
    f.set_source(source);
    return f;
}

template<class Signature>
inline function<Signature> make_function_from_builtin(const std::string &name)
{
    return function<Signature>(name);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_FUNCTION_HPP
