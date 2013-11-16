//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_CONFIG_HPP
#define BOOST_COMPUTE_CONFIG_HPP

#include <boost/config.hpp>

// the BOOST_COMPUTE_DETAIL_NO_VARIADIC_TEMPLATES macro is defined
// if the compiler does not *fully* support variadic templates
#if defined(BOOST_NO_VARIADIC_TEMPLATES) || \
    (defined(__GNUC__) && !defined(__clang__) && \
     __GNUC__ == 4 && __GNUC_MINOR__ <= 6)
  #define BOOST_COMPUTE_DETAIL_NO_VARIADIC_TEMPLATES
#endif

// the BOOST_COMPUTE_DETAIL_NO_STD_TUPLE macro is defined if the
// compiler/stdlib does not support std::tuple
#if defined(BOOST_NO_CXX11_HDR_TUPLE) || \
    defined(BOOST_NO_0X_HDR_TUPLE) || \
    defined(BOOST_COMPUTE_DETAIL_NO_VARIADIC_TEMPLATES)
  #define BOOST_COMPUTE_DETAIL_NO_STD_TUPLE
#endif

#endif // BOOST_COMPUTE_CONFIG_HPP
