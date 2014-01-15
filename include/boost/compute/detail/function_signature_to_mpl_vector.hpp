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
#include <boost/preprocessor/repetition.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/adapted/boost_tuple.hpp>

#include <boost/compute/config.hpp>

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

#define BOOST_COMPUTE_FUNCTION_SIGNATURE_TO_MPL_VECTOR_ARG_TYPE(z, n, unused) \
    typedef typename BOOST_PP_CAT(BOOST_PP_CAT(traits::arg, BOOST_PP_INC(n)), _type) BOOST_PP_CAT(T, n);

#define BOOST_COMPUTE_FUNCTION_SIGNATURE_TO_MPL_VECTOR_IMPL(z, n, unused) \
template<class Signature> \
struct function_signature_to_mpl_vector_impl<Signature, n> \
{ \
    typedef typename boost::function_traits<Signature> traits; \
    BOOST_PP_REPEAT(n, BOOST_COMPUTE_FUNCTION_SIGNATURE_TO_MPL_VECTOR_ARG_TYPE, ~) \
    typedef mpl::vector<BOOST_PP_ENUM_PARAMS(n, T)> type; \
};

BOOST_PP_REPEAT_FROM_TO(1, BOOST_COMPUTE_MAX_ARITY, BOOST_COMPUTE_FUNCTION_SIGNATURE_TO_MPL_VECTOR_IMPL, ~)

#undef BOOST_COMPUTE_FUNCTION_SIGNATURE_TO_MPL_VECTOR_IMPL

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
