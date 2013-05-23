//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_TUPLE_HPP
#define BOOST_COMPUTE_TUPLE_HPP

#include <string>
#include <utility>

#include <boost/tuple/tuple.hpp>

#include <boost/compute/functional/get.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/meta_kernel.hpp>

namespace boost {
namespace compute {
namespace detail {

// meta_kernel operators for boost::tuple literals
template<class T1>
inline meta_kernel&
operator<<(meta_kernel &kernel, const boost::tuple<T1> &x)
{
    kernel << "(" << type_name<boost::tuple<T1> >() << ")"
           << "{" << kernel.make_lit(boost::get<0>(x)) << "}";

    return kernel;
}

template<class T1, class T2>
inline meta_kernel&
operator<<(meta_kernel &kernel, const boost::tuple<T1, T2> &x)
{
    kernel << "(" << type_name<boost::tuple<T1, T2> >() << ")"
           << "{" << kernel.make_lit(boost::get<0>(x)) << ", "
                  << kernel.make_lit(boost::get<1>(x)) << "}";

    return kernel;
}

template<class T1, class T2, class T3>
inline meta_kernel&
operator<<(meta_kernel &kernel, const boost::tuple<T1, T2, T3> &x)
{
    kernel << "(" << type_name<boost::tuple<T1, T2, T3> >() << ")"
           << "{" << kernel.make_lit(boost::get<0>(x)) << ", "
                  << kernel.make_lit(boost::get<1>(x)) << ", "
                  << kernel.make_lit(boost::get<2>(x)) << "}";

    return kernel;
}

// inject_type() specializations for boost::tuple
template<class T1>
struct inject_type_impl<boost::tuple<T1> >
{
    void operator()(meta_kernel &kernel)
    {
        typedef boost::tuple<T1> tuple_type;

        kernel.inject_type<T1>();

        std::stringstream declaration;
        declaration << "typedef struct {\n"
                    << "    " << type_name<T1>() << " v0;\n"
                    << "} " << type_name<tuple_type>() << ";\n";

        kernel.add_type_declaration<tuple_type>(declaration.str());
    }
};

template<class T1, class T2>
struct inject_type_impl<boost::tuple<T1, T2> >
{
    void operator()(meta_kernel &kernel)
    {
        typedef boost::tuple<T1, T2> tuple_type;

        kernel.inject_type<T1>();
        kernel.inject_type<T2>();

        std::stringstream declaration;
        declaration << "typedef struct {\n"
                    << "    " << type_name<T1>() << " v0;\n"
                    << "    " << type_name<T2>() << " v1;\n"
                    << "} " << type_name<tuple_type>() << ";\n";

        kernel.add_type_declaration<tuple_type>(declaration.str());
    }
};

template<class T1, class T2, class T3>
struct inject_type_impl<boost::tuple<T1, T2, T3> >
{
    void operator()(meta_kernel &kernel)
    {
        typedef boost::tuple<T1, T2, T3> tuple_type;

        kernel.inject_type<T1>();
        kernel.inject_type<T2>();
        kernel.inject_type<T3>();

        std::stringstream declaration;
        declaration << "typedef struct {\n"
                    << "    " << type_name<T1>() << " v0;\n"
                    << "    " << type_name<T2>() << " v1;\n"
                    << "    " << type_name<T3>() << " v2;\n"
                    << "} " << type_name<tuple_type>() << ";\n";

        kernel.add_type_declaration<tuple_type>(declaration.str());
    }
};

// type_name() specializations for boost::tuple
template<class T1>
struct type_name_trait<boost::tuple<T1> >
{
    static const char* value()
    {
        static std::string name =
            std::string("_tuple_") + type_name<T1>() + "_t";

        return name.c_str();
    }
};

template<class T1, class T2>
struct type_name_trait<boost::tuple<T1, T2> >
{
    static const char* value()
    {
        static std::string name =
            std::string("_tuple_") +
            type_name<T1>() + "_" +
            type_name<T2>() + "_t";

        return name.c_str();
    }
};

template<class T1, class T2, class T3>
struct type_name_trait<boost::tuple<T1, T2, T3> >
{
    static const char* value()
    {
        static std::string name =
            std::string("_tuple_") +
            type_name<T1>() + "_" +
            type_name<T2>() + "_" +
            type_name<T3>() + "_t";

        return name.c_str();
    }
};

// get<N>() result type specialization for boost::tuple<>
template<size_t N, class T1>
struct get_result_type<N, boost::tuple<T1> >
{
    typedef typename boost::tuple<T1> T;

    typedef typename boost::tuples::element<N, T>::type type;
};

template<size_t N, class T1, class T2>
struct get_result_type<N, boost::tuple<T1, T2> >
{
    typedef typename boost::tuple<T1, T2> T;

    typedef typename boost::tuples::element<N, T>::type type;
};

template<size_t N, class T1, class T2, class T3>
struct get_result_type<N, boost::tuple<T1, T2, T3> >
{
    typedef typename boost::tuple<T1, T2, T3> T;

    typedef typename boost::tuples::element<N, T>::type type;
};

// get<N>() specialization for boost::tuple<>
template<size_t N, class Arg, class T1>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const invoked_get<N, Arg, boost::tuple<T1> > &expr)
{
    typedef typename boost::tuple<T1> T;

    BOOST_STATIC_ASSERT(N < size_t(boost::tuples::length<T>::value));

    kernel.inject_type<T>();

    return kernel << expr.m_arg << ".v" << uint_(N);
}

template<size_t N, class Arg, class T1, class T2>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const invoked_get<N, Arg, boost::tuple<T1, T2> > &expr)
{
    typedef typename boost::tuple<T1, T2> T;

    BOOST_STATIC_ASSERT(N < size_t(boost::tuples::length<T>::value));

    kernel.inject_type<T>();

    return kernel << expr.m_arg << ".v" << uint_(N);
}

template<size_t N, class Arg, class T1, class T2, class T3>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const invoked_get<N, Arg, boost::tuple<T1, T2, T3> > &expr)
{
    typedef typename boost::tuple<T1, T2, T3> T;

    BOOST_STATIC_ASSERT(N < size_t(boost::tuples::length<T>::value));

    kernel.inject_type<T>();

    return kernel << expr.m_arg << ".v" << uint_(N);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_TUPLE_HPP
