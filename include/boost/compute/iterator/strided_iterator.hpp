//---------------------------------------------------------------------------//
// Copyright (c) 2015 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ITERATOR_strided_ITERATOR_HPP
#define BOOST_COMPUTE_ITERATOR_strided_ITERATOR_HPP

#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include <boost/compute/functional.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>
#include <boost/compute/detail/read_write_single_value.hpp>
#include <boost/compute/iterator/detail/get_base_iterator_buffer.hpp>
#include <boost/compute/type_traits/is_device_iterator.hpp>
#include <boost/compute/type_traits/result_of.hpp>

namespace boost {
namespace compute {

// forward declaration for strided_iterator
template<class Iterator>
class strided_iterator;

namespace detail {

// helper class which defines the iterator_adaptor super-class
// type for strided_iterator
template<class Iterator>
class strided_iterator_base
{
public:
    typedef ::boost::iterator_adaptor<
        ::boost::compute::strided_iterator<Iterator>,
        Iterator
    > type;
};

// helper class for including stride value in index
// expression
template<class IndexExpr, class Stride>
struct stride_expr
{
    stride_expr(const IndexExpr &expr, Stride stride)
        : m_index_expr(expr),
          m_stride(stride)
    {
    }

    IndexExpr m_index_expr;
    Stride m_stride;
};

template<class IndexExpr, class Stride>
inline stride_expr<IndexExpr, Stride> make_stride_expr(const IndexExpr &expr, Stride stride)
{
    return stride_expr<IndexExpr, Stride>(expr, stride);
}

template<class IndexExpr, class Stride>
inline meta_kernel& operator<<(meta_kernel &kernel, const stride_expr<IndexExpr, Stride> &expr)
{
    return kernel << "(" << kernel.lit<uint_>(expr.m_stride) << " * (" << expr.m_index_expr << "))";
}

template<class Iterator, class Stride, class IndexExpr>
struct strided_iterator_index_expr
{
    typedef typename std::iterator_traits<Iterator>::value_type result_type;

    strided_iterator_index_expr(const Iterator &input_iter,
                                  const Stride &stride,
                                  const IndexExpr &index_expr)
        : m_input_iter(input_iter),
          m_stride(stride),
          m_index_expr(index_expr)
    {
    }

    Iterator m_input_iter;
    const Stride& m_stride;
    IndexExpr m_index_expr;
};

template<class Iterator, class Stride, class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const strided_iterator_index_expr<Iterator,
                                                                   Stride,
                                                                   IndexExpr> &expr)
{
    return kernel << expr.m_input_iter[make_stride_expr(expr.m_index_expr, expr.m_stride)];
}

} // end detail namespace

/// \class strided_iterator
/// \brief Iterator adaptor which skips over multiple elements each time it is incremented.
///
/// TODO
///
///
/// \see buffer_iterator, make_strided_iterator()
template<class Iterator>
class strided_iterator :
    public detail::strided_iterator_base<Iterator>::type
{
public:
    typedef typename
        detail::strided_iterator_base<Iterator>::type super_type;
    typedef typename super_type::value_type value_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::base_type base_type;
    typedef typename super_type::difference_type difference_type;

    strided_iterator(Iterator iterator, difference_type stride)
        : super_type(iterator),
          m_stride(static_cast<difference_type>(stride))
    {
        // stride must be greater than zero
        BOOST_ASSERT_MSG(stride > 0, "Stride value must be greater than zero");
    }

    strided_iterator(const strided_iterator<Iterator> &other)
        : super_type(other.base()),
          m_stride(other.m_stride)
    {
    }

    strided_iterator<Iterator>&
    operator=(const strided_iterator<Iterator> &other)
    {
        if(this != &other){
            super_type::operator=(other);

            m_stride = other.m_stride;
        }

        return *this;
    }

    ~strided_iterator()
    {
    }

    size_t get_index() const
    {
        return super_type::base().get_index();
    }

    const buffer& get_buffer() const
    {
        return detail::get_base_iterator_buffer(*this);
    }

    template<class IndexExpression>
    detail::strided_iterator_index_expr<Iterator, difference_type, IndexExpression>
    operator[](const IndexExpression &expr) const
    {
        return detail::strided_iterator_index_expr<Iterator,
                                                   difference_type,
                                                   IndexExpression>(super_type::base(),
                                                                    m_stride,
                                                                    expr);
    }

private:
    friend class ::boost::iterator_core_access;

    reference dereference() const
    {
        return reference();
    }

    bool equal(const strided_iterator<Iterator> &other) const
    {
        return (other.m_stride == m_stride)
                   && (other.base_reference() == this->base_reference());
    }

    void increment()
    {
        std::advance(super_type::base_reference(), m_stride);
    }

    void decrement()
    {
        std::advance(super_type::base_reference(),-m_stride);
    }

    void advance(typename super_type::difference_type n)
    {
        std::advance(super_type::base_reference(), n * m_stride);
    }

    difference_type distance_to(const strided_iterator<Iterator> &other) const
    {
        return std::distance(this->base_reference(), other.base_reference()) / m_stride;
    }

private:
    difference_type m_stride;
};

/// Returns a strided_iterator for \p iterator with \p stride.
///
/// TODO: Better description.
///
/// \param iterator
/// \param stride
///
/// \return a \c strided_iterator for \p iterator with \p stride.
///
/// For example, to create an iterator which iterates over every second
/// value in a \c vector<int>:
/// \code
/// auto strided_iterator = make_strided_iterator(vec.begin(), 2);
/// \endcode
template<class Iterator>
inline strided_iterator<Iterator>
make_strided_iterator(Iterator iterator, typename std::iterator_traits<Iterator>::difference_type stride)
{
    return strided_iterator<Iterator>(iterator, stride);
}

/// \internal_ (is_device_iterator specialization for transform_iterator)
template<class Iterator>
struct is_device_iterator<strided_iterator<Iterator> > : boost::true_type {};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ITERATOR_TRANSFORM_ITERATOR_HPP
