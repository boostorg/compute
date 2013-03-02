//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_IS_CONTIGUOUS_ITERATOR_HPP
#define BOOST_COMPUTE_DETAIL_IS_CONTIGUOUS_ITERATOR_HPP

#include <vector>
#include <valarray>

#include <boost/config.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

namespace boost {
namespace compute {
namespace detail {

// default = false
template<class Iterator, class Enable = void>
struct is_contiguous_iterator : public boost::false_type {};

// std::vector<T>::iterator = true
template<class Iterator>
struct is_contiguous_iterator<
    Iterator,
    typename boost::enable_if<
        typename boost::is_same<
            typename std::vector<typename Iterator::value_type>::iterator,
            typename boost::remove_const<Iterator>::type
        >::type
    >::type
> : public boost::true_type {};

// std::vector<T>::const_iterator = true
template<class Iterator>
struct is_contiguous_iterator<
    Iterator,
    typename boost::enable_if<
        typename boost::is_same<
            typename std::vector<typename Iterator::value_type>::const_iterator,
            typename boost::remove_const<Iterator>::type
        >::type
    >::type
> : public boost::true_type {};

// std::valarray<T>::iterator = true
template<class Iterator>
struct is_contiguous_iterator<
    Iterator,
    typename boost::enable_if<
        typename boost::is_same<
            typename std::valarray<typename Iterator::value_type>::iterator,
            typename boost::remove_const<Iterator>::type
        >::type
    >::type
> : public boost::true_type {};

// std::valarray<T>::const_iterator = true
template<class Iterator>
struct is_contiguous_iterator<
    Iterator,
    typename boost::enable_if<
        typename boost::is_same<
            typename std::valarray<typename Iterator::value_type>::const_iterator,
            typename boost::remove_const<Iterator>::type
        >::type
    >::type
> : public boost::true_type {};

// T* = true
template<class Iterator>
struct is_contiguous_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_pointer<Iterator>
    >::type
> : public boost::true_type {};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_IS_CONTIGUOUS_ITERATOR_HPP
