//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_BUFFER_HPP
#define BOOST_COMPUTE_BUFFER_HPP

#include <boost/move/move.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/memory_object.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

/// \class buffer
/// \brief A memory buffer on a compute device.
///
/// The buffer class represents a memory buffer on a compute device.
///
/// \see vector
class buffer : public memory_object
{
public:
    /// Creates a null buffer object.
    buffer()
        : memory_object()
    {
    }

    /// Creates a buffer object for \p mem. If \p retain is \c true, the
    /// reference count for \p mem will be incremented.
    explicit buffer(cl_mem mem, bool retain = true)
        : memory_object(mem, retain)
    {
    }

    /// Create a new memory buffer in of \p size with \p flags in
    /// \p context.
    ///
    /// \see_opencl_ref{clCreateBuffer}
    buffer(const context &context,
           size_t size,
           cl_mem_flags flags = read_write,
           void *host_ptr = 0)
    {
        cl_int error = 0;
        m_mem = clCreateBuffer(context,
                               flags,
                               (std::max)(size, size_t(1)),
                               host_ptr,
                               &error);
        if(!m_mem){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }
    }

    /// Creates a new buffer object as a copy of \p other.
    buffer(const buffer &other)
        : memory_object(other)
    {
    }

    buffer(BOOST_RV_REF(buffer) other)
        : memory_object(boost::move(static_cast<memory_object &>(other)))
    {
    }

    buffer& operator=(const buffer &other)
    {
        if(this != &other){
            memory_object::operator=(other);
        }

        return *this;
    }

    buffer& operator=(BOOST_RV_REF(buffer) other)
    {
        if(this != &other){
            memory_object::operator=(
                boost::move(static_cast<memory_object &>(other))
            );
        }

        return *this;
    }

    /// Destroys the buffer object.
    ~buffer()
    {
    }

    /// Returns the size of the buffer in bytes.
    size_t size() const
    {
        return get_memory_size();
    }

    /// \internal_
    size_t max_size() const
    {
        return get_context().get_device().max_memory_alloc_size();
    }

    /// Returns information about the buffer.
    ///
    /// \see_opencl_ref{clGetMemObjectInfo}
    template<class T>
    T get_info(cl_mem_info info) const
    {
        return get_memory_info<T>(info);
    }

private:
    BOOST_COPYABLE_AND_MOVABLE(buffer)
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_BUFFER_HPP
