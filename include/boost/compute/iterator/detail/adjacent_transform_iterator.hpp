//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ITERATOR_DETAIL_ADJACENT_TRANSFORM_ITERATOR_HPP
#define BOOST_COMPUTE_ITERATOR_DETAIL_ADJACENT_TRANSFORM_ITERATOR_HPP

#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>
#include <boost/compute/detail/is_device_iterator.hpp>
#include <boost/compute/iterator/detail/get_base_iterator_buffer.hpp>

namespace boost {
namespace compute {
namespace detail {

// forward declaration for transform_iterator
template<class InputIterator, class BinaryFunction>
class adjacent_transform_iterator;

// meta-function returning the value_type for an adjacent_transform_iterator
template<class InputIterator, class BinaryFunction>
struct make_adjacent_transform_iterator_value_type
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    typedef typename
        boost::tr1_result_of<BinaryFunction(value_type, value_type)>::type
        type;
};

// helper class which defines the iterator_adaptor super-class
// type for adjacent_transform_iterator
template<class InputIterator, class BinaryFunction>
class adjacent_transform_iterator_base
{
public:
    typedef ::boost::iterator_adaptor<
        adjacent_transform_iterator<InputIterator, BinaryFunction>,
        InputIterator,
        typename make_adjacent_transform_iterator_value_type<InputIterator, BinaryFunction>::type,
        typename std::iterator_traits<InputIterator>::iterator_category,
        typename make_adjacent_transform_iterator_value_type<InputIterator, BinaryFunction>::type
    > type;
};

template<class IndexExpr>
struct index_minus_one
{
    index_minus_one(const IndexExpr &expr)
        : m_expr(expr)
    {
    }

    IndexExpr m_expr;
};

template<class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const index_minus_one<IndexExpr> &expr)
{
    return kernel << expr.m_expr << "-1";
}

template<class InputIterator, class BinaryFunction, class IndexExpr>
struct adjacent_transform_iterator_index_expr
{
    typedef typename
        make_adjacent_transform_iterator_value_type<
            InputIterator,
            BinaryFunction
        >::type result_type;

    adjacent_transform_iterator_index_expr(const InputIterator &input_iter,
                                           const BinaryFunction &transform_expr,
                                           const IndexExpr &index_expr)
        : m_input_iter(input_iter),
          m_transform_expr(transform_expr),
          m_index_expr(index_expr)
    {
    }

    InputIterator m_input_iter;
    BinaryFunction m_transform_expr;
    IndexExpr m_index_expr;
};

template<class InputIterator, class BinaryFunction, class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const adjacent_transform_iterator_index_expr<InputIterator,
                                                                            BinaryFunction,
                                                                            IndexExpr> &expr)
{
    typedef typename std::iterator_traits<InputIterator>::value_type input_type;

    IndexExpr index_expr = expr.m_index_expr;
    index_minus_one<IndexExpr> index_expr_minus_one(index_expr);

    kernel << "(" << index_expr << " == 0 ? "
           << expr.m_transform_expr(expr.m_input_iter[index_expr],
                                    kernel.expr<input_type>("0"))
           << " : "
           << expr.m_transform_expr(expr.m_input_iter[index_expr],
                                    expr.m_input_iter[index_expr_minus_one])
           << ")";

    return kernel;
}

template<class InputIterator, class BinaryFunction>
class adjacent_transform_iterator :
    public adjacent_transform_iterator_base<InputIterator, BinaryFunction>::type
{
public:
    typedef typename
        adjacent_transform_iterator_base<InputIterator, BinaryFunction>::type
        super_type;
    typedef typename super_type::value_type value_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::base_type base_type;
    typedef typename super_type::difference_type difference_type;
    typedef BinaryFunction binary_function;

    adjacent_transform_iterator(InputIterator iterator, BinaryFunction transform)
        : super_type(iterator),
          m_transform(transform)
    {
    }

    adjacent_transform_iterator(const adjacent_transform_iterator<InputIterator,
                                                                  BinaryFunction> &other)
        : super_type(other.base()),
          m_transform(other.m_transform)
    {
    }

    adjacent_transform_iterator<InputIterator, BinaryFunction>&
    operator=(const adjacent_transform_iterator<InputIterator,
                                                BinaryFunction> &other)
    {
        if(this != &other){
            super_type::operator=(other);

            m_transform = other.m_transform;
        }

        return *this;
    }

    ~adjacent_transform_iterator()
    {
    }

    size_t get_index() const
    {
        return super_type::base().get_index();
    }

    const buffer& get_buffer() const
    {
        return get_base_iterator_buffer(*this);
    }

    template<class IndexExpression>
    adjacent_transform_iterator_index_expr<
        InputIterator,
        BinaryFunction,
        IndexExpression
    >
    operator[](const IndexExpression &expr) const
    {
        return adjacent_transform_iterator_index_expr<
                   InputIterator,
                   BinaryFunction,
                   IndexExpression
               >(super_type::base(), m_transform, expr);
    }

private:
    friend class ::boost::iterator_core_access;

    reference dereference() const
    {
        return reference();
    }

private:
    BinaryFunction m_transform;
};

template<class InputIterator, class BinaryFunction>
inline adjacent_transform_iterator<InputIterator, BinaryFunction>
make_adjacent_transform_iterator(InputIterator iterator, BinaryFunction transform)
{
    return adjacent_transform_iterator<InputIterator,
                                       BinaryFunction>(iterator, transform);
}

// is_device_iterator specialization for adjacent_transform_iterator
template<class Iterator>
struct is_device_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            adjacent_transform_iterator<typename Iterator::base_type,
                                        typename Iterator::binary_function>,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ITERATOR_DETAIL_ADJACENT_TRANSFORM_ITERATOR_HPP
