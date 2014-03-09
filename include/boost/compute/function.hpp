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
#include <sstream>

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/type_traits/function_traits.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/function_signature_to_mpl_vector.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class ResultType, class ArgTuple>
class invoked_function
{
public:
    typedef ResultType result_type;

    BOOST_STATIC_CONSTANT(
        size_t, arity = boost::tuples::length<ArgTuple>::value
    );

    invoked_function(const std::string &name, const std::string &source)
        : m_name(name),
          m_source(source)
    {
    }

    invoked_function(const std::string &name,
                     const std::string &source,
                     const ArgTuple &args)
        : m_name(name),
          m_source(source),
          m_args(args)
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

    const ArgTuple& args() const
    {
        return m_args;
    }

private:
    std::string m_name;
    std::string m_source;
    ArgTuple m_args;
};

} // end detail namespace

/// \class function
/// \brief A function object.
template<class Signature>
class function
{
public:
    /// \internal_
    typedef typename
        boost::function_traits<Signature>::result_type result_type;

    /// \internal_
    BOOST_STATIC_CONSTANT(
        size_t, arity = boost::function_traits<Signature>::arity
    );

    /// Creates a new function object with \p name.
    function(const std::string &name)
        : m_name(name)
    {
    }

    /// Destroys the function object.
    ~function()
    {
    }

    /// \internal_
    std::string name() const
    {
        return m_name;
    }

    /// \internal_
    void set_source(const std::string &source)
    {
        m_source = source;
    }

    /// \internal_
    std::string source() const
    {
        return m_source;
    }

    /// \internal_
    detail::invoked_function<result_type, boost::tuple<> >
    operator()() const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 0,
            "Non-nullary function invoked with zero arguments"
        );

        return detail::invoked_function<result_type, boost::tuple<> >(
            m_name, m_source
        );
    }

    /// \internal_
    template<class Arg1>
    detail::invoked_function<result_type, boost::tuple<Arg1> >
    operator()(const Arg1 &arg1) const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 1,
            "Non-unary function invoked one argument"
        );

        return detail::invoked_function<result_type, boost::tuple<Arg1> >(
            m_name, m_source, boost::make_tuple(arg1)
        );
    }

    /// \internal_
    template<class Arg1, class Arg2>
    detail::invoked_function<result_type, boost::tuple<Arg1, Arg2> >
    operator()(const Arg1 &arg1, const Arg2 &arg2) const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 2,
            "Non-binary function invoked with two arguments"
        );

        return detail::invoked_function<result_type, boost::tuple<Arg1, Arg2> >(
            m_name, m_source, boost::make_tuple(arg1, arg2)
        );
    }

    /// \internal_
    template<class Arg1, class Arg2, class Arg3>
    detail::invoked_function<result_type, boost::tuple<Arg1, Arg2, Arg3> >
    operator()(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3) const
    {
        BOOST_STATIC_ASSERT_MSG(
            arity == 3,
            "Non-ternary function invoked with two arguments"
        );

        return detail::invoked_function<result_type, boost::tuple<Arg1, Arg2, Arg3> >(
            m_name, m_source, boost::make_tuple(arg1, arg2, arg3)
        );
    }

private:
    std::string m_name;
    std::string m_source;
};

/// Creates a function object given its \p name and \p source.
///
/// \param name The function name.
/// \param source The function source code.
///
/// \see BOOST_COMPUTE_FUNCTION()
template<class Signature>
inline function<Signature>
make_function_from_source(const std::string &name, const std::string &source)
{
    function<Signature> f(name);
    f.set_source(source);
    return f;
}

namespace detail {

struct signature_argument_inserter
{
    signature_argument_inserter(std::stringstream &s_, size_t last)
        : s(s_)
    {
        n = 0;
        m_last = last;
    }

    template<class T>
    void operator()(const T&)
    {
        s << type_name<T>() << " _" << n+1;
        if(n+1 < m_last){
            s << ", ";
        }
        n++;
    }

    size_t n;
    size_t m_last;
    std::stringstream &s;
};

template<class Signature>
std::string make_function_declaration(const std::string &name)
{
    typedef typename
        boost::function_traits<Signature>::result_type result_type;
    typedef typename
        function_signature_to_mpl_vector<Signature>::type signature_vector;
    typedef typename
        mpl::size<signature_vector>::type arity_type;

    std::stringstream s;
    signature_argument_inserter i(s, arity_type::value);
    s << "inline " << type_name<result_type>() << " " << name;
    s << "(";
    mpl::for_each<signature_vector>(i);
    s << ")";
    return s.str();
}

// used by the BOOST_COMPUTE_FUNCTION() macro to create a function
// with the given signature, name, and source.
template<class Signature>
inline function<Signature>
make_function_impl(const std::string &name, const std::string &source)
{
    std::stringstream s;
    s << make_function_declaration<Signature>(name);
    s << source;
    return make_function_from_source<Signature>(name, s.str());
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

/// Creates a function object with \p name and \p source.
///
/// \param return_type The return type for the function.
/// \param name The name of the function.
/// \param args A list of argument types for the function.
/// \param source The OpenCL C source code for the function.
///
/// The function declaration and signature are automatically created using
/// the \p return_type, \p name, and \p args macro parameters. The argument
/// names are \c _1 to \c _N (where \c N is the arity of the function).
///
/// The source code for the function is interpreted as OpenCL C99 source code
/// which is stringified and passed to the OpenCL compiler when the function
/// is invoked.
///
/// For example, to create a function which squares a number:
/// \code
/// BOOST_COMPUTE_FUNCTION(float, square, (float),
/// {
///     return _1 * _1;
/// });
/// \endcode
///
/// And to create a function which sums two numbers:
/// \code
/// BOOST_COMPUTE_FUNCTION(int, sum_two, (int, int),
/// {
///     return _1 + _2;
/// });
/// \endcode
///
/// \see BOOST_COMPUTE_CLOSURE()
#ifdef BOOST_COMPUTE_DOXYGEN_INVOKED
#define BOOST_COMPUTE_FUNCTION(return_type, name, args, source)
#else
#define BOOST_COMPUTE_FUNCTION(return_type, name, args, ...) \
    ::boost::compute::function<return_type args> name = \
        ::boost::compute::detail::make_function_impl<return_type args>( \
            #name, #__VA_ARGS__ \
        )
#endif

#endif // BOOST_COMPUTE_FUNCTION_HPP
