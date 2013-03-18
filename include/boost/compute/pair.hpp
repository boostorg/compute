//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_PAIR_HPP
#define BOOST_COMPUTE_PAIR_HPP

#include <string>
#include <utility>

#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/meta_kernel.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class T1, class T2>
meta_kernel& operator<<(meta_kernel &kernel, const std::pair<T1, T2> &x)
{
    kernel << "(" << type_name<std::pair<T1, T2> >() << ")"
           << "{" << x.first << ", " << x.second << "}";

    return kernel;
}

template<size_t N, class T1, class T2>
struct make_pair_get_result_type;

template<class T1, class T2>
struct make_pair_get_result_type<0, T1, T2>
{
    typedef T1 type;
};

template<class T1, class T2>
struct make_pair_get_result_type<1, T1, T2>
{
    typedef T2 type;
};

template<size_t N, class T1, class T2, class Arg>
struct invoked_pair_get
{
    typedef typename make_pair_get_result_type<N, T1, T2>::type result_type;

    invoked_pair_get(const Arg &arg)
        : m_arg(arg)
    {
    }

    Arg m_arg;
};

template<size_t N, class T1, class T2, class Arg>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const invoked_pair_get<N, T1, T2, Arg> &expr)
{
    kernel.inject_type<std::pair<T1, T2> >();

    return kernel << expr.m_arg << (N == 0 ? ".first" : ".second");
}

// inject_type() specialization for std::pair
template<class T1, class T2>
struct inject_type_impl<std::pair<T1, T2> >
{
    void operator()(meta_kernel &kernel)
    {
        typedef std::pair<T1, T2> pair_type;

        kernel.inject_type<T1>();
        kernel.inject_type<T2>();

        std::stringstream declaration;
        declaration << "typedef struct {\n"
                    << "    " << type_name<T1>() << " first;\n"
                    << "    " << type_name<T2>() << " second;\n"
                    << "} " << type_name<pair_type>() << ";\n";

        kernel.add_type_declaration<pair_type>(declaration.str());
    }
};

} // end detail namespace

template<size_t N, class T1, class T2>
struct get_pair
{
    typedef typename detail::make_pair_get_result_type<N, T1, T2>::type result_type;

    template<class Arg>
    detail::invoked_pair_get<N, T1, T2, Arg>
    operator()(const Arg &x) const
    {
        return detail::invoked_pair_get<N, T1, T2, Arg>(x);
    }
};

namespace detail {

// type_name() specialization for std::pair
template<class T1, class T2>
struct type_name_trait<std::pair<T1, T2> >
{
    static const char* value()
    {
        static std::string name =
            std::string("_pair_") +
            type_name<T1>() + "_" + type_name<T2>() +
            "_t";

        return name.c_str();
    }
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_PAIR_HPP
