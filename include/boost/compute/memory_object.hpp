//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_MEMORY_OBJECT_HPP
#define BOOST_COMPUTE_MEMORY_OBJECT_HPP

#include <boost/compute/cl.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

class memory_object
{
public:
    enum mem_flags {
        read_write = CL_MEM_READ_WRITE,
        read_only = CL_MEM_READ_ONLY,
        write_only = CL_MEM_WRITE_ONLY,
        use_host_ptr = CL_MEM_USE_HOST_PTR,
        alloc_host_ptr = CL_MEM_ALLOC_HOST_PTR,
        copy_host_ptr = CL_MEM_COPY_HOST_PTR
    };

    cl_mem get_mem() const
    {
        return m_mem;
    }

    size_t get_memory_size() const
    {
        return get_memory_info<size_t>(CL_MEM_SIZE);
    }

    cl_mem_object_type get_memory_type() const
    {
        return get_memory_info<cl_mem_object_type>(CL_MEM_TYPE);
    }

    cl_mem_flags get_memory_flags() const
    {
        return get_memory_info<cl_mem_flags>(CL_MEM_FLAGS);
    }

    context get_context() const
    {
        return context(get_memory_info<cl_context>(CL_MEM_CONTEXT));
    }

    void* get_host_ptr() const
    {
        return get_memory_info<void *>(CL_MEM_HOST_PTR);
    }

    template<class T>
    T get_memory_info(cl_mem_info info) const
    {
        return detail::get_object_info<T>(clGetMemObjectInfo, m_mem, info);
    }

    bool operator==(const memory_object &other) const
    {
        return m_mem == other.m_mem;
    }

    bool operator!=(const memory_object &other) const
    {
        return m_mem != other.m_mem;
    }

protected:
    memory_object(const cl_mem &mem = cl_mem())
        : m_mem(mem)
    {
        if(m_mem){
            clRetainMemObject(m_mem);
        }
    }

    memory_object(const memory_object &other)
        : m_mem(other.m_mem)
    {
        if(m_mem){
            clRetainMemObject(m_mem);
        }
    }

    #if !defined(BOOST_NO_RVALUE_REFERENCES)
    memory_object(memory_object &&other)
        : m_mem(other.m_mem)
    {
        other.m_mem = 0;
    }
    #endif // !defined(BOOST_NO_RVALUE_REFERENCES)

    memory_object& operator=(const memory_object &other)
    {
        if(this != &other){
            if(m_mem){
                clReleaseMemObject(m_mem);
            }

            m_mem = other.m_mem;

            if(m_mem){
                clRetainMemObject(m_mem);
            }
        }

        return *this;
    }

    ~memory_object()
    {
        if(m_mem){
            clReleaseMemObject(m_mem);
        }
    }

protected:
    cl_mem m_mem;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_MEMORY_OBJECT_HPP
