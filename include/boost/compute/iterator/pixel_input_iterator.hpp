//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ITERATOR_PIXEL_INPUT_ITERATOR_HPP
#define BOOST_COMPUTE_ITERATOR_PIXEL_INPUT_ITERATOR_HPP

#include <cstddef>
#include <iterator>

#include <boost/config.hpp>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <boost/compute/image2d.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/image_sampler.hpp>
#include <boost/compute/type_traits/make_vector_type.hpp>
#include <boost/compute/detail/is_device_iterator.hpp>

namespace boost {
namespace compute {

// forward declaration for pixel_input_iterator<T>
template<class T> class pixel_input_iterator;

namespace detail {

// helper class which defines the iterator_facade super-class
// type for pixel_input_iterator<T>
template<class T>
class pixel_input_iterator_base
{
public:
    typedef ::boost::iterator_facade<
        ::boost::compute::pixel_input_iterator<T>,
        typename ::boost::compute::make_vector_type<T, 4>::type,
        ::std::random_access_iterator_tag
    > type;
};

template<class T, class IndexExpr>
struct pixel_input_iterator_index_expr
{
    typedef typename make_vector_type<T, 4>::type result_type;

    pixel_input_iterator_index_expr(const image2d &image, const IndexExpr &expr)
        : m_image(image),
          m_expr(expr)
    {
    }

    const image2d &m_image;
    IndexExpr m_expr;
};

template<class T, class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const pixel_input_iterator_index_expr<T, IndexExpr> &expr)
{
    std::string image =
        kernel.get_image_identifier("__read_only", expr.m_image);
    std::string sampler =
        kernel.get_sampler_identifier(false,
                                      image_sampler::none,
                                      image_sampler::nearest);

    std::string read_suffix;
    if(boost::is_floating_point<T>::value){
        read_suffix = "f";
    }
    else if(boost::is_unsigned<T>::value){
        read_suffix = "ui";
    }
    else {
        read_suffix = "i";
    }

    return kernel << "read_image" << read_suffix << "("
                  << image << ","
                  << sampler << ","
                  << "(int2)("
                  << expr.m_expr << " % get_image_width(" << image << "), "
                  << expr.m_expr << " / get_image_width(" << image << ")))";
}

} // end detail namespace

template<class T>
class pixel_input_iterator : public detail::pixel_input_iterator_base<T>::type
{
public:
    typedef typename detail::pixel_input_iterator_base<T>::type super_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::difference_type difference_type;

    pixel_input_iterator()
        : m_image(0),
          m_index(0)
    {
    }

    pixel_input_iterator(const image2d &image, size_t index)
        : m_image(&image),
          m_index(index)
    {
    }

    pixel_input_iterator(const pixel_input_iterator<T> &other)
        : m_image(other.m_image),
          m_index(other.m_index)
    {
    }

    pixel_input_iterator<T>& operator=(const pixel_input_iterator<T> &other)
    {
        if(this != &other){
            m_image = other.m_image;
            m_index = other.m_index;
        }

        return *this;
    }

    ~pixel_input_iterator()
    {
    }

    const image2d& get_image() const
    {
        return *m_image;
    }

    size_t get_index() const
    {
        return m_index;
    }

    template<class Expr>
    detail::pixel_input_iterator_index_expr<T, Expr>
    operator[](const Expr &expr) const
    {
        return detail::pixel_input_iterator_index_expr<T, Expr>(*m_image, expr);
    }

private:
    friend class ::boost::iterator_core_access;

    reference dereference() const
    {
        return T(0);
    }

    bool equal(const pixel_input_iterator<T> &other) const
    {
        return m_image == other.m_image && m_index == other.m_index;
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

    difference_type distance_to(const pixel_input_iterator<T> &other) const
    {
        return static_cast<difference_type>(other.m_index - m_index);
    }

private:
    const image2d *m_image;
    size_t m_index;
};

template<class T>
inline pixel_input_iterator<T>
make_pixel_input_iterator(const image2d &image, size_t index = 0)
{
    return pixel_input_iterator<T>(image, index);
}

namespace detail {

// is_device_iterator specialization for pixel_input_iterator
template<class Iterator>
struct is_device_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            pixel_input_iterator<typename Iterator::value_type>,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

} // end detail namespace

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ITERATOR_PIXEL_INPUT_ITERATOR_HPP
