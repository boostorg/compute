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

#ifdef BOOST_COMPUTE_HAVE_GL
#include <boost/compute/cl_gl.hpp>
#endif

namespace boost {
namespace compute {

class buffer : public memory_object
{
public:
    buffer()
        : memory_object()
    {
    }

    explicit buffer(cl_mem mem, bool retain = true)
        : memory_object(mem, retain)
    {
    }

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

    ~buffer()
    {
    }

    size_t size() const
    {
        return get_memory_size();
    }

    size_t max_size() const
    {
        return get_context().get_device().max_memory_alloc_size();
    }

    template<class T>
    T get_info(cl_mem_info info) const
    {
        return get_memory_info<T>(info);
    }

    #ifdef BOOST_COMPUTE_HAVE_GL
    static buffer from_gl_buffer(const context &context,
                                 GLuint bufobj,
                                 cl_mem_flags flags = read_write)
    {
        cl_int error = 0;
        cl_mem mem = clCreateFromGLBuffer(context, flags, bufobj, &error);
        if(!mem){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }

        buffer buf(mem);
        clReleaseMemObject(mem);
        return buf;
    }
    #endif // BOOST_COMPUTE_HAVE_GL

private:
    BOOST_COPYABLE_AND_MOVABLE(buffer)
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_BUFFER_HPP
