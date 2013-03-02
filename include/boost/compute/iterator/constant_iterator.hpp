//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ITERATOR_CONSTANT_ITERATOR_HPP
#define BOOST_COMPUTE_ITERATOR_CONSTANT_ITERATOR_HPP

#include <string>
#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/is_device_iterator.hpp>

namespace boost {
namespace compute {

// forward declaration for constant_iterator<T>
template<class T> class constant_iterator;

namespace detail {

// helper class which defines the iterator_facade super-class
// type for constant_iterator<T>
template<class T>
class constant_iterator_base
{
public:
    typedef ::boost::iterator_facade<
        ::boost::compute::constant_iterator<T>,
        T,
        ::std::random_access_iterator_tag
    > type;
};

} // end detail namespace

template<class T>
class constant_iterator : public detail::constant_iterator_base<T>::type
{
public:
    typedef typename detail::constant_iterator_base<T>::type super_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::difference_type difference_type;

    constant_iterator(const T &value, size_t index = 0)
        : m_value(value),
          m_index(index)
    {
    }

    constant_iterator(const constant_iterator<T> &other)
        : m_value(other.m_value),
          m_index(other.m_index)
    {
    }

    constant_iterator<T>& operator=(const constant_iterator<T> &other)
    {
        if(this != &other){
            m_value = other.m_value;
            m_index = other.m_index;
        }

        return *this;
    }

    ~constant_iterator()
    {
    }

    size_t get_index() const
    {
        return m_index;
    }

    template<class Expr>
    detail::meta_kernel_literal<T> operator[](const Expr &expr) const
    {
        (void) expr;

        return detail::meta_kernel::make_lit<T>(m_value);
    }

private:
    friend class ::boost::iterator_core_access;

    reference dereference() const
    {
        return m_value;
    }

    bool equal(const constant_iterator<T> &other) const
    {
        return m_value == other.m_value;
    }

    void increment()
    {
        m_index++;
    }

    void decrement()
    {
        m_index--;
    }

    void advance(difference_type n)
    {
        m_index = static_cast<size_t>(static_cast<difference_type>(m_index) + n);
    }

    difference_type distance_to(const constant_iterator<T> &other) const
    {
        return static_cast<difference_type>(other.m_index - m_index);
    }

private:
    T m_value;
    size_t m_index;
};

template<class T>
inline constant_iterator<T>
make_constant_iterator(const T &value, size_t index = 0)
{
    return constant_iterator<T>(value, index);
}

namespace detail {

// is_device_iterator specialization for constant_iterator
template<class Iterator>
struct is_device_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            constant_iterator<typename Iterator::value_type>,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

} // end detail namespace

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ITERATOR_CONSTANT_ITERATOR_HPP
