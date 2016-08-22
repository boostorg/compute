//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_VECTOR_HPP
#define BOOST_COMPUTE_DISTRIBUTED_VECTOR_HPP

#include <algorithm>
#include <vector>
#include <cstddef>
#include <iterator>
#include <exception>

#include <boost/throw_exception.hpp>

#include <boost/compute/config.hpp>

#ifndef BOOST_COMPUTE_NO_HDR_INITIALIZER_LIST
#include <initializer_list>
#endif

#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/allocator/buffer_allocator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/type_traits/detail/capture_traits.hpp>
#include <boost/compute/detail/buffer_value.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>
#include <boost/compute/distributed/copy.hpp>
#include <boost/compute/distributed/detail/weight_func.hpp>

namespace boost {
namespace compute {
namespace distributed {

/// \class vector
/// \brief A resizable array of values allocated across multiple devices.
///
/// The vector<T> class stores a dynamic array of values. Internally, the data
/// is stored in OpenCL buffer objects from multiple OpenCL contexts.
template<
    class T,
    weight_func weight = default_weight_func,
    class Alloc = ::boost::compute::buffer_allocator<T>
>
class vector
{
public:
    typedef T value_type;
    typedef Alloc allocator_type;
    typedef typename allocator_type::size_type size_type;
    typedef typename allocator_type::difference_type difference_type;
    typedef ::boost::compute::detail::buffer_value<T> reference;
    typedef const ::boost::compute::detail::buffer_value<T> const_reference;
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef buffer_iterator<T> iterator;
    typedef buffer_iterator<T> const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    /// Creates an empty distributed vector using \p queue.
    explicit vector(const command_queue &queue)
        : m_queue(queue),
          m_size(0)
    {
        for(size_t i = 0; i < m_queue.size(); i++)
        {
            m_allocators.push_back(Alloc(m_queue.get_context(i)));
            m_data.push_back(
                m_allocators.back()
                    .allocate(_minimum_capacity())
            );
            m_data_sizes.push_back(0);
            m_data_indices.push_back(0);
        }
    }

    /// Creates a distributed vector with space for \p count elements
    /// in \p context.
    explicit vector(size_t count, const command_queue &queue)
        : m_queue(queue),
          m_size(count)
    {
        allocate_memory(count);
    }

    /// Creates a distributed  vector with space for \p count elements and
    /// sets each equal to \p value.
    ///
    /// For example:
    /// \code
    /// // creates a vector with four values set to nine (e.g. [9, 9, 9, 9]).
    /// boost::compute::distributed::vector<int> vec(4, 9, queue);
    /// \endcode
    vector(size_type count,
           const T &value,
           command_queue &queue)
        : m_queue(queue),
          m_size(count)
    {
        allocate_memory(m_size);
        std::vector<event> events;
        events.reserve(m_data.size());
        for(size_t i = 0; i < m_data.size(); i++)
        {
            event e =
                ::boost::compute::fill_async(
                    begin(i),
                    end(i),
                    value,
                    queue.get(i)
                ).get_event();
            if(e.get()) {
                events.push_back(e);
            }
        }
        for(size_t i = 0; i < events.size(); i++) {
            events[i].wait();
        }
    }

    /// Creates a vector with space for the values in the range [\p first,
    /// \p last) allocated on the host and copies them into the vector
    /// with \p queue.
    ///
    /// For example:
    /// \code
    /// // values on the host
    /// int data[] = { 1, 2, 3, 4 };
    ///
    /// // create a vector of size four and copy the values from data
    /// boost::compute::distributed::vector<int> vec(data, data + 4, queue);
    /// \endcode
    template<class HostIterator>
    vector(HostIterator first,
           HostIterator last,
           command_queue &queue,
           typename boost::enable_if_c<
               !is_device_iterator<HostIterator>::value
           >::type* = 0)
        : m_queue(queue),
          m_size(::boost::compute::detail::iterator_range_size(first, last))
    {
        allocate_memory(m_size);
        ::boost::compute::distributed::copy(first, last, *this, m_queue);
    }

    /// Creates a vector with space for the values in the range [\p first,
    /// \p last) allocated on an OpenCL device and copies them into the vector
    /// with \p queue.
    template<class DeviceIterator>
    vector(DeviceIterator first,
           DeviceIterator last,
           ::boost::compute::command_queue &device_queue,
           command_queue &distributed_queue,
           typename boost::enable_if_c<
               is_device_iterator<DeviceIterator>::value
           >::type* = 0)
        : m_queue(distributed_queue),
          m_size(::boost::compute::detail::iterator_range_size(first, last))
    {
        allocate_memory(m_size);
        ::boost::compute::distributed::copy(
            first, last, *this, device_queue, m_queue
        );
    }

    /// Creates a new vector and copies the values from \p other.
    explicit vector(const vector &other)
        : m_queue(other.m_queue),
          m_size(other.m_size)
    {
        allocate_memory(m_size);
        ::boost::compute::distributed::copy(
            other, *this, m_queue
        );
    }

    /// Creates a new vector and copies the values from \p other
    /// with \p queue.
    vector(const vector &other, command_queue &queue)
        : m_queue(queue),
          m_size(other.m_size)
    {
        allocate_memory(m_size);
        if(m_queue == other.m_queue) {
            ::boost::compute::distributed::copy(
                other, *this, m_queue
            );
        }
        else {
            command_queue other_queue = other.get_queue();
            ::boost::compute::distributed::copy(
                other, *this, other_queue, m_queue
            );
        }
    }

    /// Creates a new vector and copies the values from \p other
    /// with \p queue.
    template<class OtherAlloc>
    vector(const vector<T, weight, OtherAlloc> &other)
        : m_queue(other.m_queue),
          m_size(other.m_size)
    {
        allocate_memory(m_size);
        ::boost::compute::distributed::copy(
            other, *this, m_queue
        );
    }

    /// Creates a new vector and copies the values from \p other.
    template<class OtherAlloc>
    vector(const vector<T, weight, OtherAlloc> &other,
           command_queue &queue)
        : m_queue(queue),
          m_size(other.size())
    {
        allocate_memory(m_size);
        if(m_queue == other.get_queue()) {
            ::boost::compute::distributed::copy(
                other, *this, m_queue
            );
        }
        else {
            command_queue other_queue = other.get_queue();
            ::boost::compute::distributed::copy(
                other, *this, other_queue, m_queue
            );
        }
    }

    /// Creates a new vector and copies the values from \p vector.
    template<class OtherAlloc>
    vector(const std::vector<T, OtherAlloc> &vector,
           command_queue &queue)
        : m_queue(queue),
          m_size(vector.size())
    {
        allocate_memory(m_size);
        ::boost::compute::distributed::copy(
            vector.begin(), vector.end(), *this, m_queue
        );
    }

    /// Copy assignment. This operation is always non-blocking.
    vector& operator=(const vector &other)
    {
        if(this != &other){
            m_queue = other.m_queue;
            m_size = other.m_size;
            allocate_memory(m_size);
            ::boost::compute::distributed::copy(other, *this, m_queue);
        }
        return *this;
    }

    /// Copy assignment. This operation is always non-blocking.
    template<class OtherAlloc>
    vector& operator=(const vector<T, weight, OtherAlloc> &other)
    {
        m_queue = other.get_queue();
        m_size = other.size();
        allocate_memory(m_size);
        ::boost::compute::distributed::copy(other, *this, m_queue);
        return *this;
    }

    /// Copy assignment. This operation is always non-blocking.
    template<class OtherAlloc>
    vector& operator=(const std::vector<T, OtherAlloc> &vector)
    {
        m_size = vector.size();
        allocate_memory(m_size);
        ::boost::compute::distributed::copy(
            vector.begin(), vector.end(), *this, m_queue
        );
        return *this;
    }

    #ifndef BOOST_COMPUTE_NO_RVALUE_REFERENCES
    /// Move-constructs a new vector from \p other.
    vector(vector&& other)
        : m_queue(std::move(m_queue)),
          m_size(other.m_size),
          m_data(std::move(other.m_data)),
          m_data_sizes(std::move(other.m_data_sizes)),
          m_data_indices(std::move(other.m_data_indices)),
          m_allocators(std::move(other.m_allocators))
    {
        other.m_size = 0;
    }

    /// Move-assigns the data from \p other to \c *this.
    vector& operator=(vector&& other)
    {
        if(m_size) {
            for(size_t i = 0; i < m_allocators.size(); i++) {
                m_allocators[i].deallocate(m_data[i], m_data_sizes[i]);
            }
        }

        m_queue = std::move(other.m_queue);
        m_size = other.m_size;
        m_data = std::move(other.m_data);
        m_data_sizes = std::move(other.m_data_sizes);
        m_data_indices = std::move(other.m_data_indices);
        m_allocators = std::move(other.m_allocators);

        other.m_size = 0;

        return *this;
    }
    #endif // BOOST_COMPUTE_NO_RVALUE_REFERENCES

    /// Destroys the vector object.
    ~vector()
    {
        if(m_size) {
            for(size_t i = 0; i < m_allocators.size(); i++) {
                m_allocators[i].deallocate(m_data[i], m_data_sizes[i]);
            }
        }
    }

    /// Returns the number of elements in the vector.
    size_type size() const
    {
        return m_size;
    }

    size_t parts() const
    {
        return m_data.size();
    }

    std::vector<size_type> parts_sizes() const
    {
        return m_data_sizes;
    }

    size_type part_size(size_t n) const
    {
        return m_data_sizes[n];
    }

    std::vector<size_t> parts_starts() const
    {
        return m_data_indices;
    }

    size_t part_start(size_t n) const
    {
        return m_data_indices[n];
    }

    /// Returns \c true if the vector is empty.
    bool empty() const
    {
        return m_size == 0;
    }

    iterator begin(size_t n)
    {
        return ::boost::compute::make_buffer_iterator<T>(
            m_data[n].get_buffer(), 0
        );
    }

    const_iterator begin(size_t n) const
    {
        return ::boost::compute::make_buffer_iterator<T>(
            m_data[n].get_buffer(), 0
        );
    }

    const_iterator cbegin(size_t n) const
    {
        return begin(n);
    }

    iterator end(size_t n)
    {
        return ::boost::compute::make_buffer_iterator<T>(
            m_data[n].get_buffer(), m_data_sizes[n]
        );
    }

    const_iterator end(size_t n) const
    {
        return ::boost::compute::make_buffer_iterator<T>(
            m_data[n].get_buffer(), m_data_sizes[n]
        );
    }

    const_iterator cend(size_t n) const
    {
        return end(n);
    }

    reverse_iterator rbegin(size_t n)
    {
        return reverse_iterator(end(n) - 1);
    }

    const_reverse_iterator rbegin(size_t n) const
    {
        return reverse_iterator(end(n) - 1);
    }

    const_reverse_iterator crbegin(size_t n) const
    {
        return rbegin(n);
    }

    reverse_iterator rend(size_t n)
    {
        return reverse_iterator(begin(n) - 1);
    }

    const_reverse_iterator rend(size_t n) const
    {
        return reverse_iterator(begin(n) - 1);
    }

    const_reverse_iterator crend(size_t n) const
    {
        return rend(n);
    }

    reference operator[](size_type index)
    {
        size_t n =
            std::upper_bound(m_data_indices.begin(), m_data_indices.end(), index)
                - m_data_indices.begin() - 1;
        size_t part_index = index - m_data_indices[n];
        return *(begin(n) + static_cast<difference_type>(part_index));
    }

    const_reference operator[](size_type index) const
    {
        size_t n =
            std::upper_bound(m_data_indices.begin(), m_data_indices.end(), index)
                - m_data_indices.begin() - 1;
        size_t part_index = index - m_data_indices[n];
        return *(begin(n) + static_cast<difference_type>(part_index));
    }

    reference at(size_type index)
    {
        if(index >= size()){
            BOOST_THROW_EXCEPTION(std::out_of_range("index out of range"));
        }

        return operator[](index);
    }

    const_reference at(size_type index) const
    {
        if(index >= size()){
            BOOST_THROW_EXCEPTION(std::out_of_range("index out of range"));
        }

        return operator[](index);
    }

    reference front()
    {
        return *begin(parts() - 1);
    }

    const_reference front() const
    {
        return *begin(parts() - 1);
    }

    reference back()
    {
        return *(end(parts() - 1) - static_cast<difference_type>(1));
    }

    const_reference back() const
    {
        return *(end(parts() - 1) - static_cast<difference_type>(1));
    }

    /// Swaps the contents of \c *this with \p other.
    void swap(vector &other)
    {
        std::swap(m_data, other.m_data);
        std::swap(m_data_sizes, other.m_data_sizes);
        std::swap(m_data_indices, other.m_data_indices);
        std::swap(m_size, other.m_size);
        std::swap(m_allocators, other.m_allocators);
        std::swap(m_queue, other.m_queue);
    }

    /// Removes all elements from the vector.
    void clear()
    {
        m_size = 0;
    }

    /// Returns the underlying buffer.
    std::vector<buffer> get_buffers() const
    {
        std::vector<buffer> buffers;
        buffers.reserve(m_data.size());
        for(size_t i = 0; i < m_data.size(); i++) {
            buffers.push_back(m_data[i].get_buffer());
        }
        return buffers;
    }

    /// Returns the underlying buffer for part \p n.
    const buffer& get_buffer(size_t n) const
    {
        return m_data[n].get_buffer();
    }

    command_queue get_queue() const
    {
        return m_queue;
    }

    /// command queue.
    const context& get_context() const
    {
        return m_queue.get_context();
    }

private:
    /// \internal_
    BOOST_CONSTEXPR size_type _align() const { return 16; }

    /// \internal_
    BOOST_CONSTEXPR size_type _minimum_capacity() const { return _align(); }

    /// \internal_
    BOOST_CONSTEXPR float _growth_factor() const { return 1.5; }

    void allocate_memory(size_type count)
    {
        m_allocators.clear();
        m_data.clear();
        m_data_sizes.clear();
        m_data_indices.clear();

        m_allocators.reserve(m_queue.size());
        m_data.reserve(m_queue.size());
        m_data_sizes.reserve(m_queue.size());
        m_data_indices.reserve(m_queue.size());

        std::vector<size_t> partition =
            detail::partition(m_queue, weight, count, _align());
        for(size_t i = 0; i < m_queue.size(); i++)
        {
            size_type data_size = partition[i + 1] - partition[i];
            m_allocators.push_back(Alloc(m_queue.get_context(i)));
            m_data.push_back(
                m_allocators.back()
                    .allocate((std::max)(data_size, _minimum_capacity()))
                );
            m_data_sizes.push_back(data_size);
            m_data_indices.push_back(partition[i]);
        }
    }

private:
    command_queue m_queue;
    size_type m_size;

    std::vector<size_type> m_data_sizes;
    std::vector<size_t> m_data_indices;
    std::vector<pointer> m_data;
    std::vector<allocator_type> m_allocators;
};

} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_VECTOR_HPP */
