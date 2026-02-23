//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_DETAIL_IS_DISTRIBUTED_VECTOR_HPP
#define BOOST_COMPUTE_DISTRIBUTED_DETAIL_IS_DISTRIBUTED_VECTOR_HPP


#include <boost/type_traits/integral_constant.hpp>

namespace boost {
namespace compute {
namespace distributed {

namespace detail {

template<class T>
struct is_distributed_vector : boost::false_type {};

template<class T>
struct is_distributed_vector<const T> : is_distributed_vector<T> {};

} // end detail namespace

} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_DETAIL_IS_DISTRIBUTED_VECTOR_HPP */
