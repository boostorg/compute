//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_CONTAINER_FLAT_SET_HPP
#define BOOST_COMPUTE_CONTAINER_FLAT_SET_HPP

#include <cstddef>
#include <utility>

#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/algorithm/lower_bound.hpp>
#include <boost/compute/algorithm/upper_bound.hpp>
#include <boost/compute/container/vector.hpp>

namespace boost {
namespace compute {

template<class T>
class flat_set
{
public:
    typedef T key_type;
    typedef typename vector<T>::value_type value_type;
    typedef typename vector<T>::size_type size_type;
    typedef typename vector<T>::difference_type difference_type;
    typedef typename vector<T>::reference reference;
    typedef typename vector<T>::const_reference const_reference;
    typedef typename vector<T>::pointer pointer;
    typedef typename vector<T>::const_pointer const_pointer;
    typedef typename vector<T>::iterator iterator;
    typedef typename vector<T>::const_iterator const_iterator;
    typedef typename vector<T>::reverse_iterator reverse_iterator;
    typedef typename vector<T>::const_reverse_iterator const_reverse_iterator;

    explicit flat_set(const context &context = system::default_context())
        : m_vector(context)
    {
    }

    flat_set(const flat_set<T> &other)
        : m_vector(other.m_vector)
    {
    }

    flat_set<T>& operator=(const flat_set<T> &other)
    {
        if(this != &other){
            m_vector = other.m_vector;
        }

        return *this;
    }

    ~flat_set()
    {
    }

    iterator begin()
    {
        return m_vector.begin();
    }

    const_iterator begin() const
    {
        return m_vector.begin();
    }

    const_iterator cbegin() const
    {
        return m_vector.cbegin();
    }

    iterator end()
    {
        return m_vector.end();
    }

    const_iterator end() const
    {
        return m_vector.end();
    }

    const_iterator cend() const
    {
        return m_vector.cend();
    }

    reverse_iterator rbegin()
    {
        return m_vector.rbegin();
    }

    const_reverse_iterator rbegin() const
    {
        return m_vector.rbegin();
    }

    const_reverse_iterator crbegin() const
    {
        return m_vector.crbegin();
    }

    reverse_iterator rend()
    {
        return m_vector.rend();
    }

    const_reverse_iterator rend() const
    {
        return m_vector.rend();
    }

    const_reverse_iterator crend() const
    {
        return m_vector.crend();
    }

    size_type size() const
    {
        return m_vector.size();
    }

    size_type max_size() const
    {
        return m_vector.max_size();
    }

    bool empty() const
    {
        return m_vector.empty();
    }

    size_type capacity() const
    {
        return m_vector.capacity();
    }

    void reserve(size_type size)
    {
        m_vector.reserve(size);
    }

    void shrink_to_fit()
    {
        m_vector.shrink_to_fit();
    }

    void clear()
    {
        m_vector.clear();
    }

    std::pair<iterator, bool> insert(const value_type &value)
    {
        iterator location = upper_bound(value);

        if(location != begin() && *(location - 1) == value){
            return std::make_pair(location - 1, false);
        }
        else {
            m_vector.insert(location, value);
            return std::make_pair(location, true);
        }
    }

    iterator erase(const const_iterator &position)
    {
        return erase(position, position + 1);
    }

    iterator erase(const const_iterator &first, const const_iterator &last)
    {
        return m_vector.erase(first, last);
    }

    size_type erase(const key_type &value)
    {
        iterator position = find(value);

        if(position == end()){
            return 0;
        }
        else {
            erase(position);
            return 1;
        }
    }

    iterator find(const key_type &value)
    {
        return ::boost::compute::find(begin(), end(), value);
    }

    const_iterator find(const key_type &value) const
    {
        return ::boost::compute::find(begin(), end(), value);
    }

    size_type count(const key_type &value) const
    {
        return find(value) != end() ? 1 : 0;
    }

    iterator lower_bound(const key_type &value)
    {
        return ::boost::compute::lower_bound(begin(), end(), value);
    }

    const_iterator lower_bound(const key_type &value) const
    {
        return ::boost::compute::lower_bound(begin(), end(), value);
    }

    iterator upper_bound(const key_type &value)
    {
        return ::boost::compute::upper_bound(begin(), end(), value);
    }

    const_iterator upper_bound(const key_type &value) const
    {
        return ::boost::compute::upper_bound(begin(), end(), value);
    }

private:
    vector<T> m_vector;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_CONTAINER_FLAT_SET_HPP
