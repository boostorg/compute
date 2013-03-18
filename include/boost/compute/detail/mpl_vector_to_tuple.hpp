//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_MPL_VECTOR_TO_TUPLE_HPP
#define BOOST_COMPUTE_DETAIL_MPL_VECTOR_TO_TUPLE_HPP

#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/adapted/boost_tuple.hpp>

namespace boost {
namespace compute {
namespace detail {

namespace mpl = boost::mpl;

template<class Vector, size_t N>
struct mpl_vector_to_tuple_impl;

template<class Vector>
struct mpl_vector_to_tuple_impl<Vector, 1>
{
    typedef typename
        boost::tuple<
            typename mpl::at_c<Vector, 0>::type
        > type;
};

template<class Vector>
struct mpl_vector_to_tuple_impl<Vector, 2>
{
    typedef typename
        boost::tuple<
            typename mpl::at_c<Vector, 0>::type,
            typename mpl::at_c<Vector, 1>::type
        > type;
};

template<class Vector>
struct mpl_vector_to_tuple_impl<Vector, 3>
{
    typedef typename
        boost::tuple<
            typename mpl::at_c<Vector, 0>::type,
            typename mpl::at_c<Vector, 1>::type,
            typename mpl::at_c<Vector, 2>::type
        > type;
};

// meta-function which converts a mpl::vector to a boost::tuple
template<class Vector>
struct mpl_vector_to_tuple
{
    typedef typename
        mpl_vector_to_tuple_impl<
            Vector,
            mpl::size<Vector>::value
        >::type type;
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_MPL_VECTOR_TO_TUPLE_HPP
