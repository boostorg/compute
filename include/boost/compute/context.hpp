//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_CONTEXT_HPP
#define BOOST_COMPUTE_CONTEXT_HPP

#include <boost/compute/cl.hpp>
#include <boost/compute/device.hpp>

namespace boost {
namespace compute {

class context
{
public:
    context()
        : m_context(0)
    {
    }

    explicit context(const device &device,
                     const cl_context_properties *properties = 0)
    {
        BOOST_ASSERT(device.id() != 0);

        cl_device_id device_id = device.id();

        cl_int error = 0;
        m_context = clCreateContext(properties,
                                    1,
                                    &device_id,
                                    0,
                                    0,
                                    &error);
        if(!m_context){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }
    }

    explicit context(cl_context context, bool retain = true)
        : m_context(context)
    {
        if(m_context && retain){
            clRetainContext(m_context);
        }
    }

    context(const context &other)
        : m_context(other.m_context)
    {
        if(m_context){
            clRetainContext(m_context);
        }
    }

    context& operator=(const context &other)
    {
        if(this != &other){
            if(m_context){
                clReleaseContext(m_context);
            }

            m_context = other.m_context;

            if(m_context){
                clRetainContext(m_context);
            }
        }

        return *this;
    }

    ~context()
    {
        if(m_context){
            clReleaseContext(m_context);
        }
    }

    cl_context& get() const
    {
        return const_cast<cl_context &>(m_context);
    }

    device get_device() const
    {
        size_t count = 0;
        clGetContextInfo(m_context,
                         CL_CONTEXT_DEVICES,
                         0,
                         0,
                         &count);
        if(count == 0){
            return device();
        }

        cl_device_id id;
        clGetContextInfo(m_context,
                         CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id),
                         &id,
                         0);
        if(id == 0){
            return device();
        }

        return device(id);
    }

    template<class T>
    T get_info(cl_context_info info) const
    {
        return detail::get_object_info<T>(clGetContextInfo, m_context, info);
    }

    bool operator==(const context &other) const
    {
        return m_context == other.m_context;
    }

    bool operator!=(const context &other) const
    {
        return m_context != other.m_context;
    }

    operator cl_context() const
    {
        return m_context;
    }

private:
    cl_context m_context;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_CONTEXT_HPP
