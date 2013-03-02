//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ITERATOR_COUNTING_ITERATOR_HPP
#define BOOST_COMPUTE_ITERATOR_COUNTING_ITERATOR_HPP

#include <string>
#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/is_device_iterator.hpp>

namespace boost {
namespace compute {

// forward declaration for counting_iterator<T>
template<class T> class counting_iterator;

namespace detail {

// helper class which defines the iterator_facade super-class
// type for counting_iterator<T>
template<class T>
class counting_iterator_base
{
public:
    typedef ::boost::iterator_facade<
        ::boost::compute::counting_iterator<T>,
        T,
        ::std::random_access_iterator_tag
    > type;
};

template<class T, class IndexExpr>
struct counting_iterator_index_expr
{
    typedef T result_type;

    counting_iterator_index_expr(const T &init, const IndexExpr &expr)
        : m_init(init),
          m_expr(expr)
    {
    }

    const T &m_init;
    IndexExpr m_expr;
};

template<class T, class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const counting_iterator_index_expr<T, IndexExpr> &expr)
{
    return kernel << '(' << expr.m_init << '+' << expr.m_expr << ')';
}

} // end detail namespace

template<class T>
class counting_iterator : public detail::counting_iterator_base<T>::type
{
public:
    typedef typename detail::counting_iterator_base<T>::type super_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::difference_type difference_type;

    counting_iterator(const T &init)
        : m_init(init)
    {
    }

    counting_iterator(const counting_iterator<T> &other)
        : m_init(other.m_init)
    {
    }

    counting_iterator<T>& operator=(const counting_iterator<T> &other)
    {
        if(this != &other){
            m_init = other.m_init;
        }

        return *this;
    }

    ~counting_iterator()
    {
    }

    size_t get_index() const
    {
        return 0;
    }

    template<class Expr>
    detail::counting_iterator_index_expr<T, Expr>
    operator[](const Expr &expr) const
    {
        return detail::counting_iterator_index_expr<T, Expr>(m_init, expr);
    }

private:
    friend class ::boost::iterator_core_access;

    reference dereference() const
    {
        return m_init;
    }

    bool equal(const counting_iterator<T> &other) const
    {
        return m_init == other.m_init;
    }

    void increment()
    {
        m_init++;
    }

    void decrement()
    {
        m_init--;
    }

    void advance(difference_type n)
    {
        m_init += static_cast<T>(n);
    }

    difference_type distance_to(const counting_iterator<T> &other) const
    {
        return other.m_init - m_init;
    }

private:
    T m_init;
};

template<class T>
inline counting_iterator<T> make_counting_iterator(const T &init)
{
    return counting_iterator<T>(init);
}

namespace detail {

// is_device_iterator specialization for counting_iterator
template<class Iterator>
struct is_device_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            counting_iterator<typename Iterator::value_type>,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

} // end detail namespace

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ITERATOR_COUNTING_ITERATOR_HPP
