//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_LAMBDA_FUNCTIONAL_HPP
#define BOOST_COMPUTE_LAMBDA_FUNCTIONAL_HPP

#include <boost/tuple/tuple.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/proto/core.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/compute/functional/get.hpp>
#include <boost/compute/lambda/result_of.hpp>

namespace boost {
namespace compute {
namespace lambda {

namespace mpl = boost::mpl;
namespace proto = boost::proto;

// wraps a unary boolean function
#define BOOST_COMPUTE_LAMBDA_WRAP_BOOLEAN_UNARY_FUNCTION(name) \
    namespace detail { \
        struct BOOST_PP_CAT(name, _func) \
        { \
            template<class Expr, class Args> \
            struct lambda_result \
            { \
                typedef int type; \
            }; \
            \
            static const char* function_name() \
            { \
                return BOOST_PP_STRINGIZE(name); \
            } \
        }; \
    } \
    template<class Arg> \
    typename proto::result_of::make_expr< \
               proto::tag::function, \
               BOOST_PP_CAT(detail::name, _func), \
               const Arg& \
         >::type const \
    name(const Arg &arg) \
    { \
        return proto::make_expr<proto::tag::function>( \
                   BOOST_PP_CAT(detail::name, _func)(), \
                   ::boost::ref(arg) \
           ); \
    }

// wraps a unary function who's return type is the same as the argument type
#define BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(name) \
    namespace detail { \
        struct BOOST_PP_CAT(name, _func) \
        { \
            template<class Expr, class Args> \
            struct lambda_result \
            { \
                typedef typename proto::result_of::child_c<Expr, 1>::type Arg1; \
                typedef typename ::boost::compute::lambda::result_of<Arg1, Args>::type type; \
            }; \
            \
            static const char* function_name() \
            { \
                return BOOST_PP_STRINGIZE(name); \
            } \
        }; \
    } \
    template<class Arg> \
    typename proto::result_of::make_expr< \
               proto::tag::function, \
               BOOST_PP_CAT(detail::name, _func), \
               const Arg& \
         >::type const \
    name(const Arg &arg) \
    { \
        return proto::make_expr<proto::tag::function>( \
                   BOOST_PP_CAT(detail::name, _func)(), \
                   ::boost::ref(arg) \
           ); \
    }

#define BOOST_COMPUTE_LAMBDA_WRAP_BINARY_FUNCTION(name) \
    namespace detail { \
        struct BOOST_PP_CAT(name, _func) \
        { \
            template<class Expr, class Args> \
            struct lambda_result \
            { \
                typedef typename proto::result_of::child_c<Expr, 1>::type Arg1; \
                typedef typename ::boost::compute::lambda::result_of<Arg1, Args>::type type; \
            }; \
            \
            static const char* function_name() \
            { \
                return BOOST_PP_STRINGIZE(name); \
            } \
        }; \
    } \
    template<class Arg1, class Arg2> \
    typename proto::result_of::make_expr< \
                 proto::tag::function, \
                 BOOST_PP_CAT(detail::name, _func), \
                 const Arg1&, \
                 const Arg2& \
             >::type const \
    name(const Arg1 &arg1, const Arg2 &arg2) \
    { \
        return proto::make_expr<proto::tag::function>( \
                   BOOST_PP_CAT(detail::name, _func)(), \
                   ::boost::ref(arg1), \
                   ::boost::ref(arg2)); \
    }

// wraps a ternary function
#define BOOST_COMPUTE_LAMBDA_WRAP_TERNARY_FUNCTION(name) \
    namespace detail { \
        struct BOOST_PP_CAT(name, _func) \
        { \
            template<class Expr, class Args> \
            struct lambda_result \
            { \
                typedef typename proto::result_of::child_c<Expr, 1>::type Arg1; \
                typedef typename ::boost::compute::lambda::result_of<Arg1, Args>::type type; \
            }; \
            \
            static const char* function_name() \
            { \
                return BOOST_PP_STRINGIZE(name); \
            } \
        }; \
    } \
    template<class Arg1, class Arg2, class Arg3> \
    typename proto::result_of::make_expr< \
                 proto::tag::function, \
                 BOOST_PP_CAT(detail::name, _func), \
                 const Arg1&, \
                 const Arg2&, \
                 const Arg3& \
             >::type const \
    name(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3) \
    { \
        return proto::make_expr<proto::tag::function>( \
                   BOOST_PP_CAT(detail::name, _func)(), \
                   ::boost::ref(arg1), \
                   ::boost::ref(arg2), \
                   ::boost::ref(arg3)); \
    }


BOOST_COMPUTE_LAMBDA_WRAP_BOOLEAN_UNARY_FUNCTION(all)
BOOST_COMPUTE_LAMBDA_WRAP_BOOLEAN_UNARY_FUNCTION(any)
BOOST_COMPUTE_LAMBDA_WRAP_BOOLEAN_UNARY_FUNCTION(isinf)
BOOST_COMPUTE_LAMBDA_WRAP_BOOLEAN_UNARY_FUNCTION(isnan)
BOOST_COMPUTE_LAMBDA_WRAP_BOOLEAN_UNARY_FUNCTION(isfinite)

BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(abs)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(cos)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(acos)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(sin)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(asin)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(tan)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(atan)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(sqrt)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(rsqrt)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(exp)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(exp2)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(exp10)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(log)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(log2)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(log10)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(round)
BOOST_COMPUTE_LAMBDA_WRAP_UNARY_FUNCTION_T(length)

//BOOST_COMPUTE_LAMBDA_WRAP_BINARY_FUNCTION(cross)
//BOOST_COMPUTE_LAMBDA_WRAP_BINARY_FUNCTION(dot)
//BOOST_COMPUTE_LAMBDA_WRAP_BINARY_FUNCTION(distance)
BOOST_COMPUTE_LAMBDA_WRAP_BINARY_FUNCTION(pow)
BOOST_COMPUTE_LAMBDA_WRAP_BINARY_FUNCTION(pown)
BOOST_COMPUTE_LAMBDA_WRAP_BINARY_FUNCTION(powr)

BOOST_COMPUTE_LAMBDA_WRAP_TERNARY_FUNCTION(clamp)
BOOST_COMPUTE_LAMBDA_WRAP_TERNARY_FUNCTION(fma)
BOOST_COMPUTE_LAMBDA_WRAP_TERNARY_FUNCTION(mad)
BOOST_COMPUTE_LAMBDA_WRAP_TERNARY_FUNCTION(smoothstep)

namespace detail {

struct dot_func
{
    template<class Expr, class Args>
    struct lambda_result
    {
        typedef typename proto::result_of::child_c<Expr, 1>::type Arg1;
        typedef typename proto::result_of::child_c<Expr, 2>::type Arg2;

        typedef typename ::boost::compute::lambda::result_of<Arg1, Args>::type T1;
        typedef typename ::boost::compute::lambda::result_of<Arg2, Args>::type T2;

        typedef typename ::boost::compute::scalar_type<T1>::type type;
    };

    static const char* function_name()
    {
        return "dot";
    }
};

// function wrapper for get<N>() in lambda expressions
template<size_t N>
struct get_func
{
    template<class Expr, class Args>
    struct lambda_result
    {
        typedef typename proto::result_of::child_c<Expr, 1>::type Arg;
        typedef typename ::boost::compute::lambda::result_of<Arg, Args>::type T;
        typedef typename ::boost::compute::detail::get_result_type<N, T>::type type;
    };
};

// returns the suffix string for get<N>() in lambda expressions
// (e.g. ".x" for get<0>() with float4)
template<size_t N, class T>
struct get_func_suffix
{
    static std::string value()
    {
        BOOST_STATIC_ASSERT(N < 16);

        std::stringstream stream;

        if(N < 10){
            stream << ".s" << uint_(N);
        }
        else if(N < 16){
            stream << ".s" << char('a' + (N - 10));
        }

        return stream.str();
    }
};

template<size_t N, class T1, class T2>
struct get_func_suffix<N, std::pair<T1, T2> >
{
    static std::string value()
    {
        BOOST_STATIC_ASSERT(N < 2);

        if(N == 0){
            return ".first";
        }
        else {
            return ".second";
        }
    };
};

template<size_t N, class T1>
struct get_func_suffix<N, boost::tuple<T1> >
{
    static std::string value()
    {
        BOOST_STATIC_ASSERT(N < 1);

        return ".v" + boost::lexical_cast<std::string>(N);
    }
};

template<size_t N, class T1, class T2>
struct get_func_suffix<N, boost::tuple<T1, T2> >
{
    static std::string value()
    {
        BOOST_STATIC_ASSERT(N < 2);

        return ".v" + boost::lexical_cast<std::string>(N);
    }
};

template<size_t N, class T1, class T2, class T3>
struct get_func_suffix<N, boost::tuple<T1, T2, T3> >
{
    static std::string value()
    {
        BOOST_STATIC_ASSERT(N < 3);

        return ".v" + boost::lexical_cast<std::string>(N);
    }
};

} // end detail namespace

template<class Arg1, class Arg2>
typename proto::result_of::make_expr<
             proto::tag::function,
             detail::dot_func,
             const Arg1&,
             const Arg2&
         >::type const
dot(const Arg1 &arg1, const Arg2 &arg2)
{
    return proto::make_expr<proto::tag::function>(
               detail::dot_func(),
               ::boost::ref(arg1),
               ::boost::ref(arg2)
           );
}

// get<N>()
template<size_t N, class Arg>
typename proto::result_of::make_expr<
             proto::tag::function,
             detail::get_func<N>,
             const Arg&
         >::type const
get(const Arg &arg)
{
    return proto::make_expr<proto::tag::function>(
               detail::get_func<N>(),
               ::boost::ref(arg)
           );
}

} // end lambda namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_LAMBDA_FUNCTIONAL_HPP
