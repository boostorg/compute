//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_FUNCTION_SIGNATURE_TO_MPL_VECTOR_HPP
#define BOOST_COMPUTE_DETAIL_FUNCTION_SIGNATURE_TO_MPL_VECTOR_HPP

#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/adapted/boost_tuple.hpp>

namespace boost {
namespace compute {
namespace detail {

namespace mpl = boost::mpl;

template<class Signature, size_t Arity>
struct function_signature_to_mpl_vector_impl;

template<class Signature>
struct function_signature_to_mpl_vector_impl<Signature, 0>
{
    typedef mpl::vector<> type;
};

template<class Signature>
struct function_signature_to_mpl_vector_impl<Signature, 1>
{
    typedef typename boost::function_traits<Signature> traits;
    typedef typename traits::arg1_type T1;

    typedef mpl::vector<T1> type;
};

template<class Signature>
struct function_signature_to_mpl_vector_impl<Signature, 2>
{
    typedef typename boost::function_traits<Signature> traits;
    typedef typename traits::arg1_type T1;
    typedef typename traits::arg2_type T2;

    typedef mpl::vector<T1, T2> type;
};

template<class Signature>
struct function_signature_to_mpl_vector_impl<Signature, 3>
{
    typedef typename boost::function_traits<Signature> traits;
    typedef typename traits::arg1_type T1;
    typedef typename traits::arg2_type T2;
    typedef typename traits::arg2_type T3;

    typedef mpl::vector<T1, T2, T3> type;
};

// meta-function returning the argument types from the function
// signature as a mpl vector
template<class Signature>
struct function_signature_to_mpl_vector
{
    typedef typename function_signature_to_mpl_vector_impl<
        Signature, boost::function_traits<Signature>::arity
    >::type type;
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_FUNCTION_SIGNATURE_TO_MPL_VECTOR_HPP
