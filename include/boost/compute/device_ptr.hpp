//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DEVICE_PTR_HPP
#define BOOST_COMPUTE_DEVICE_PTR_HPP

#include <boost/config.hpp>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/detail/is_buffer_iterator.hpp>
#include <boost/compute/detail/is_device_iterator.hpp>
#include <boost/compute/detail/read_write_single_value.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class T, class IndexExpr>
struct device_ptr_index_expr
{
    typedef T result_type;

    device_ptr_index_expr(const buffer &buffer,
                          uint_ index,
                          const IndexExpr &expr)
        : m_buffer(buffer),
          m_index(index),
          m_expr(expr)
    {
    }

    operator T() const
    {
        BOOST_STATIC_ASSERT_MSG(boost::is_integral<IndexExpr>::value,
                                "Index expression must be integral");

        const context &context = m_buffer.get_context();
        const device &device = context.get_device();
        command_queue queue(context, device);

        return detail::read_single_value<T>(m_buffer, m_expr, queue);
    }

    const buffer &m_buffer;
    uint_ m_index;
    IndexExpr m_expr;
};

} // end detail namespace

template<class T>
class device_ptr
{
public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::random_access_iterator_tag iterator_category;
    typedef T* pointer;
    typedef T& reference;

    device_ptr()
        : m_buffer(0),
          m_index(0)
    {
    }

    device_ptr(const buffer &buffer, size_t index = 0)
        : m_buffer(new ::boost::compute::buffer(buffer)),
          m_index(index)
    {
    }

    device_ptr(const device_ptr<T> &other)
        : m_buffer(other.m_buffer),
          m_index(other.m_index)
    {
    }

    device_ptr<T>& operator=(const device_ptr<T> &other)
    {
        if(this != &other){
            m_buffer = other.m_buffer;
            m_index = other.m_index;
        }

        return *this;
    }

    ~device_ptr()
    {
    }

    size_type get_index() const
    {
        return m_index;
    }

    const buffer& get_buffer() const
    {
        BOOST_ASSERT(m_buffer != 0);

        return *m_buffer;
    }

    const buffer* get_buffer_ptr() const
    {
        return m_buffer;
    }

    template<class OT>
    device_ptr<OT> cast() const
    {
        return device_ptr<OT>(*m_buffer, m_index);
    }

    device_ptr<T> operator+(difference_type n) const
    {
        return device_ptr<T>(*m_buffer, m_index + n);
    }

    device_ptr<T> operator+(const device_ptr<T> &other) const
    {
        return device_ptr<T>(*m_buffer, m_index + other.m_index);
    }

    device_ptr<T>& operator+=(difference_type n)
    {
        m_index += static_cast<size_t>(n);
        return *this;
    }

    difference_type operator-(const device_ptr<T> &other) const
    {
        return static_cast<difference_type>(m_index - other.m_index);
    }

    device_ptr<T>& operator-=(difference_type n)
    {
        m_index -= n;
        return *this;
    }

    bool operator==(const device_ptr<T> &other) const
    {
        return *m_buffer == *other.m_buffer && m_index == other.m_index;
    }

    bool operator!=(const device_ptr<T> &other) const
    {
        return !(*this == other);
    }

    template<class Expr>
    detail::device_ptr_index_expr<T, Expr>
    operator[](const Expr &expr) const
    {
        BOOST_ASSERT(m_buffer);
        BOOST_ASSERT(m_buffer->get_mem());

        return detail::device_ptr_index_expr<T, Expr>(*m_buffer,
                                                      uint_(m_index),
                                                      expr);
    }

private:
    const buffer *m_buffer;
    size_t m_index;
};

namespace detail {

// is_buffer_iterator specialization for buffer_iterator
template<class Iterator>
struct is_buffer_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            device_ptr<typename Iterator::value_type>,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

// is_device_iterator specialization for buffer_iterator
template<class Iterator>
struct is_device_iterator<
    Iterator,
    typename boost::enable_if<
        boost::is_same<
            device_ptr<typename Iterator::value_type>,
            typename boost::remove_const<Iterator>::type
        >
    >::type
> : public boost::true_type {};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DEVICE_PTR_HPP
