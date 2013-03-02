//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_CONTAINER_VALARRAY_HPP
#define BOOST_COMPUTE_CONTAINER_VALARRAY_HPP

#include <cstddef>
#include <valarray>

#include <boost/compute/buffer.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/algorithm/max_element.hpp>
#include <boost/compute/algorithm/min_element.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/accumulate.hpp>
#include <boost/compute/detail/buffer_value.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace boost {
namespace compute {

template<class T>
class valarray
{
public:
    explicit valarray(const context &context = system::default_context())
        : m_buffer(context, 0)
    {
    }

    explicit valarray(size_t size,
                      const context &context = system::default_context())
        : m_buffer(context, size * sizeof(T))
    {
    }

    valarray(const T &value,
             size_t size,
             const context &context = system::default_context())
        : m_buffer(context, size * sizeof(T))
    {
        fill(begin(), end(), value);
    }

    valarray(const T *values,
             size_t size,
             const context &context = system::default_context())
        : m_buffer(context, size * sizeof(T))
    {
        copy(values, values + size, begin());
    }

    valarray(const valarray<T> &other)
        : m_buffer(other.m_buffer.get_context(), other.size() * sizeof(T))
    {
    }

    valarray(const std::valarray<T> &valarray,
             const context &context = system::default_context())
        : m_buffer(context, valarray.size() * sizeof(T))
    {
        copy(&valarray[0], &valarray[valarray.size()], begin());
    }

    valarray<T>& operator=(const valarray<T> &other)
    {
        if(this != &other){
            resize(other.size());
            copy(other.begin(), other.end(), begin());
        }

        return *this;
    }

    valarray<T>& operator=(const std::valarray<T> &valarray)
    {
        resize(valarray.size());
        copy(&valarray[0], &valarray[valarray.size()], begin());
    }

    ~valarray()
    {
    }

    size_t size() const
    {
        return m_buffer.size() / sizeof(T);
    }

    void resize(size_t size, T value = T())
    {
        m_buffer = buffer(m_buffer.get_context(), size * sizeof(T));
        fill(begin(), end(), value);
    }

    detail::buffer_value<T> operator[](size_t index)
    {
        return *(begin() + static_cast<ptrdiff_t>(index));
    }

    const detail::buffer_value<T> operator[](size_t index) const
    {
        return *(begin() + static_cast<ptrdiff_t>(index));
    }

    T (min)() const
    {
        return *(boost::compute::min_element(begin(), end()));
    }

    T (max)() const
    {
        return *(boost::compute::max_element(begin(), end()));
    }

    T sum() const
    {
        return boost::compute::accumulate(begin(), end(), T(0));
    }

    template<class UnaryFunction>
    valarray<T> apply(UnaryFunction function) const
    {
        valarray<T> result(size());
        transform(begin(), end(), result.begin(), function);
        return result;
    }

    const buffer& get_buffer() const
    {
        return m_buffer;
    }

private:
    buffer_iterator<T> begin() const
    {
        return buffer_iterator<T>(m_buffer, 0);
    }

    buffer_iterator<T> end() const
    {
        return buffer_iterator<T>(m_buffer, size());
    }

private:
    buffer m_buffer;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_CONTAINER_VALARRAY_HPP
