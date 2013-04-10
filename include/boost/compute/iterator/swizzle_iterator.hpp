//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ITERATOR_SWIZZLE_ITERATOR_HPP
#define BOOST_COMPUTE_ITERATOR_SWIZZLE_ITERATOR_HPP

#include <string>
#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include <boost/compute/types.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/type_traits/make_vector_type.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>
#include <boost/compute/detail/is_device_iterator.hpp>
#include <boost/compute/detail/read_write_single_value.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>
#include <boost/compute/iterator/detail/get_base_iterator_buffer.hpp>

namespace boost {
namespace compute {

// forward declaration for swizzle_iterator
template<class InputIterator, size_t Size>
class swizzle_iterator;

namespace detail {

// meta-function returing the value_type for a swizzle_iterator
template<class InputIterator, size_t Size>
struct make_swizzle_iterator_value_type
{
    typedef
        typename make_vector_type<
            typename scalar_type<
                typename std::iterator_traits<InputIterator>::value_type
            >::type,
            Size
        >::type type;
};

// helper class which defines the iterator_adaptor super-class
// type for swizzle_iterator
template<class InputIterator, size_t Size>
class swizzle_iterator_base
{
public:
    typedef ::boost::iterator_adaptor<
        ::boost::compute::swizzle_iterator<InputIterator, Size>,
        InputIterator,
        typename make_swizzle_iterator_value_type<InputIterator, Size>::type,
        typename std::iterator_traits<InputIterator>::iterator_category,
        typename make_swizzle_iterator_value_type<InputIterator, Size>::type
    > type;
};

template<class InputIterator, size_t Size, class IndexExpr>
struct swizzle_iterator_index_expr
{
    typedef typename make_swizzle_iterator_value_type<InputIterator, Size>::type result_type;

    swizzle_iterator_index_expr(const InputIterator &input_iter,
                                const IndexExpr &index_expr,
                                const std::string &components)
        : m_input_iter(input_iter),
          m_index_expr(index_expr),
          m_components(components)
    {
    }

    InputIterator m_input_iter;
    IndexExpr m_index_expr;
    std::string m_components;
};

template<class InputIterator, size_t Size, class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const swizzle_iterator_index_expr<InputIterator,
                                                                 Size,
                                                                 IndexExpr> &expr)
{
    return kernel << expr.m_input_iter[expr.m_index_expr]
                  << "." << expr.m_components;
}

} // end detail namespace

template<class InputIterator, size_t Size>
class swizzle_iterator :
    public detail::swizzle_iterator_base<InputIterator, Size>::type
{
public:
    typedef typename detail::swizzle_iterator_base<InputIterator, Size>::type super_type;
    typedef typename super_type::value_type value_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::base_type base_type;
    typedef typename super_type::difference_type difference_type;

    BOOST_STATIC_CONSTANT(size_t, vector_size = Size);

    swizzle_iterator(InputIterator iterator, const std::string &components)
        : super_type(iterator),
          m_components(components)
    {
        BOOST_ASSERT(components.size() == Size);
    }

    swizzle_iterator(const swizzle_iterator<InputIterator, Size> &other)
        : super_type(other.base()),
          m_components(other.m_components)
    {
        BOOST_ASSERT(m_components.size() == Size);
    }

    swizzle_iterator<InputIterator, Size>&
    operator=(const swizzle_iterator<InputIterator, Size> &other)
    {
        if(this != &other){
            super_type::operator=(other);

            m_components = other.m_components;
        }

        return *this;
    }

    ~swizzle_iterator()
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
    detail::swizzle_iterator_index_expr<InputIterator, Size, IndexExpression>
    operator[](const IndexExpression &expr) const
    {
        return detail::swizzle_iterator_index_expr<InputIterator,
                                                   Size,
                                                   IndexExpression>(super_type::base(),
                                                                    expr,
                                                                    m_components);
    }

private:
    friend class ::boost::iterator_core_access;

    reference dereference() const
    {
        return reference();
    }

private:
    std::string m_components;
};

template<size_t Size, class InputIterator>
inline swizzle_iterator<InputIterator, Size>
make_swizzle_iterator(InputIterator iterator, const std::string &components)
{
    return swizzle_iterator<InputIterator, Size>(iterator, components);
}

namespace detail {

// is_device_iterator specialization for swizzle_iterator
template<class Iterator>
struct is_device_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            swizzle_iterator<
                typename Iterator::base_type,
                Iterator::vector_size
            >,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

} // end detail namespace

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ITERATOR_SWIZZLE_ITERATOR_HPP
