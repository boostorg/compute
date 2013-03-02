//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_FUNCTIONAL_VECTOR_HPP
#define BOOST_COMPUTE_FUNCTIONAL_VECTOR_HPP

#include <boost/compute/type_traits.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class Arg, size_t N, class Result>
struct invoked_vector_component_function
{
    typedef Result result_type;

    invoked_vector_component_function(const Arg &arg)
        : m_arg(arg)
    {
    }

    Arg arg() const
    {
        return m_arg;
    }

    Arg m_arg;
};

} // end detail namespace

template<class T, size_t N>
struct vector_component
{
    typedef typename scalar_type<T>::type result_type;

    template<class Arg>
    detail::invoked_vector_component_function<Arg, N, result_type>
    operator()(const Arg &x) const
    {
        BOOST_STATIC_ASSERT(vector_size<T>::value > 1 && N < vector_size<T>::value);

        return detail::invoked_vector_component_function<Arg, N, result_type>(x);
    }
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_FUNCTIONAL_VECTOR_HPP
