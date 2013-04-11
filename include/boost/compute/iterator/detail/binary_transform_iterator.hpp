//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ITERATOR_DETAIL_BINARY_TRANSFORM_ITERATOR_HPP
#define BOOST_COMPUTE_ITERATOR_DETAIL_BINARY_TRANSFORM_ITERATOR_HPP

#include <string>
#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <boost/compute/functional.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/is_device_iterator.hpp>
#include <boost/compute/iterator/detail/get_base_iterator_buffer.hpp>

namespace boost {
namespace compute {
namespace detail {

// forward declaration for binary_transform_iterator
template<class InputIterator1, class InputIterator2, class BinaryFunction>
class binary_transform_iterator;

// meta-function returning the value_type for a binary_transform_iterator
template<class InputIterator1, class InputIterator2, class BinaryFunction>
struct make_binary_transform_iterator_value_type
{
    typedef typename std::iterator_traits<InputIterator1>::value_type value_type1;
    typedef typename std::iterator_traits<InputIterator2>::value_type value_type2;

    typedef typename
        boost::tr1_result_of<BinaryFunction(value_type1, value_type2)>::type
        type;
};

// helper class which defines the iterator_facade super-class
// type for binary_transform_iterator
template<class InputIterator1, class InputIterator2, class BinaryFunction>
class binary_transform_iterator_base
{
public:
    typedef ::boost::iterator_facade<
        binary_transform_iterator<
            InputIterator1, InputIterator2, BinaryFunction
        >,
        typename make_binary_transform_iterator_value_type<
                     InputIterator1,
                     InputIterator2,
                     BinaryFunction
                 >::type,
        typename std::iterator_traits<InputIterator1>::iterator_category,
        typename make_binary_transform_iterator_value_type<
                     InputIterator1,
                     InputIterator2,
                     BinaryFunction
                 >::type
    > type;
};

template<class InputIterator1,
         class InputIterator2,
         class BinaryFunction,
         class IndexExpr>
struct binary_transform_iterator_index_expr
{
    typedef typename
        make_binary_transform_iterator_value_type<
            InputIterator1,
            InputIterator2,
            BinaryFunction>::type result_type;

    binary_transform_iterator_index_expr(const InputIterator1 &iterator1,
                                         const InputIterator2 &iterator2,
                                         const BinaryFunction &transform_expr,
                                         const IndexExpr &index_expr)
        : m_input_iter1(iterator1),
          m_input_iter2(iterator2),
          m_transform_expr(transform_expr),
          m_index_expr(index_expr)
    {
    }

    InputIterator1 m_input_iter1;
    InputIterator2 m_input_iter2;
    BinaryFunction m_transform_expr;
    IndexExpr m_index_expr;
};

template<class InputIterator1,
         class InputIterator2,
         class BinaryFunction,
         class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const binary_transform_iterator_index_expr<
                                   InputIterator1,
                                   InputIterator2,
                                   BinaryFunction,
                                   IndexExpr
                               > &expr)
{
    return kernel << expr.m_transform_expr(expr.m_input_iter1[expr.m_index_expr],
                                           expr.m_input_iter2[expr.m_index_expr]);
}

template<class InputIterator1, class InputIterator2, class BinaryFunction>
class binary_transform_iterator :
    public binary_transform_iterator_base<
               InputIterator1, InputIterator2, BinaryFunction
           >::type
{
public:
    typedef typename
        binary_transform_iterator_base<
            InputIterator1, InputIterator2, BinaryFunction
        >::type super_type;
    typedef typename super_type::difference_type difference_type;
    typedef typename super_type::reference reference;
    typedef InputIterator1 base_type;
    typedef InputIterator1 base1_type;
    typedef InputIterator2 base2_type;
    typedef BinaryFunction binary_function;

    binary_transform_iterator(InputIterator1 iterator1,
                              InputIterator2 iterator2,
                              BinaryFunction transform)
        : m_iterator1(iterator1),
          m_iterator2(iterator2),
          m_transform(transform)
    {
    }

    binary_transform_iterator(
        const binary_transform_iterator<InputIterator1,
                                        InputIterator2,
                                        BinaryFunction> &other)
        : m_iterator1(other.m_iterator1),
          m_iterator2(other.m_iterator2),
          m_transform(other.m_transform)
    {
    }

    binary_transform_iterator<InputIterator1,
                              InputIterator2,
                              BinaryFunction>&
    operator=(const binary_transform_iterator<InputIterator1,
                                              InputIterator2,
                                              BinaryFunction> &other)
    {
        if(this != &other){
            m_iterator1 = other.m_iterator1;
            m_iterator2 = other.m_iterator2;
            m_transform = other.m_transform;
        }

        return *this;
    }

    ~binary_transform_iterator()
    {
    }

    size_t get_index() const
    {
        return m_iterator1.get_index();
    }

    const buffer& get_buffer() const
    {
        return get_base_iterator_buffer(*this);
    }

    template<class IndexExpression>
    binary_transform_iterator_index_expr<
        InputIterator1,
        InputIterator2,
        BinaryFunction,
        IndexExpression
    >
    operator[](const IndexExpression &expr) const
    {
        return binary_transform_iterator_index_expr<
                   InputIterator1,
                   InputIterator2,
                   BinaryFunction,
                   IndexExpression
               >(m_iterator1, m_iterator2, m_transform, expr);
    }

    const base_type& base() const
    {
        return m_iterator1;
    }

    const base1_type& base1() const
    {
        return m_iterator1;
    }

    const base2_type& base2() const
    {
        return m_iterator2;
    }

private:
    friend class ::boost::iterator_core_access;

    reference dereference() const
    {
        return reference();
    }

    bool equal(
        const binary_transform_iterator<InputIterator1,
                                        InputIterator2,
                                        BinaryFunction> &other) const
    {
        return m_iterator1 == other.m_iterator1 &&
               m_iterator2 == other.m_iterator2 &&
               m_transform == other.m_transform;
    }

    void increment()
    {
        m_iterator1++;
        m_iterator2++;
    }

    void decrement()
    {
        m_iterator1--;
        m_iterator2--;
    }

    void advance(difference_type n)
    {
        m_iterator1 += n;
        m_iterator2 += n;
    }

    difference_type distance_to(
        const binary_transform_iterator<InputIterator1,
                                        InputIterator2,
                                        BinaryFunction> &other) const
    {
        return std::distance(m_iterator1, other.m_iterator1);
    }

private:
    InputIterator1 m_iterator1;
    InputIterator2 m_iterator2;
    BinaryFunction m_transform;
};

template<class InputIterator1, class InputIterator2, class BinaryFunction>
inline binary_transform_iterator<InputIterator1,
                                 InputIterator2,
                                 BinaryFunction>
make_binary_transform_iterator(InputIterator1 iterator1,
                               InputIterator2 iterator2,
                               BinaryFunction transform)
{
    return binary_transform_iterator<InputIterator1,
                                     InputIterator2,
                                     BinaryFunction>(iterator1,
                                                     iterator2,
                                                     transform);
}

// is_device_iterator specialization for binary_transform_iterator
template<class Iterator>
struct is_device_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            binary_transform_iterator<typename Iterator::base1_type,
                                      typename Iterator::base2_type,
                                      typename Iterator::binary_function>,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ITERATOR_DETAIL_BINARY_TRANSFORM_ITERATOR_HPP
