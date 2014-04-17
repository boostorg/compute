//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_EVENT_HPP
#define BOOST_COMPUTE_EVENT_HPP

#include <boost/move/move.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/config.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/detail/get_object_info.hpp>
#include <boost/compute/detail/assert_cl_success.hpp>

namespace boost {
namespace compute {

/// \class event
/// \brief An event on a compute device.
class event
{
public:
    /// \internal_
    enum execution_status {
        complete = CL_COMPLETE,
        running = CL_RUNNING,
        submitted = CL_SUBMITTED,
        queued = CL_QUEUED
    };

    /// \internal_
    enum command_type {
        ndrange_kernel = CL_COMMAND_NDRANGE_KERNEL,
        task = CL_COMMAND_TASK,
        native_kernel = CL_COMMAND_NATIVE_KERNEL,
        read_buffer = CL_COMMAND_READ_BUFFER,
        write_buffer = CL_COMMAND_WRITE_BUFFER,
        copy_buffer = CL_COMMAND_COPY_BUFFER,
        read_image = CL_COMMAND_READ_IMAGE,
        write_image = CL_COMMAND_WRITE_IMAGE,
        copy_image = CL_COMMAND_COPY_IMAGE,
        copy_image_to_buffer = CL_COMMAND_COPY_IMAGE_TO_BUFFER,
        copy_buffer_to_image = CL_COMMAND_COPY_BUFFER_TO_IMAGE,
        map_buffer = CL_COMMAND_MAP_BUFFER,
        map_image = CL_COMMAND_MAP_IMAGE,
        unmap_mem_object = CL_COMMAND_UNMAP_MEM_OBJECT,
        marker = CL_COMMAND_MARKER,
        aquire_gl_objects = CL_COMMAND_ACQUIRE_GL_OBJECTS,
        release_gl_object = CL_COMMAND_RELEASE_GL_OBJECTS
        #if defined(CL_VERSION_1_1)
        ,
        read_buffer_rect = CL_COMMAND_READ_BUFFER_RECT,
        write_buffer_rect = CL_COMMAND_WRITE_BUFFER_RECT,
        copy_buffer_rect = CL_COMMAND_COPY_BUFFER_RECT
        #endif
    };

    /// \internal_
    enum profiling_info {
        profiling_command_queued = CL_PROFILING_COMMAND_QUEUED,
        profiling_command_submit = CL_PROFILING_COMMAND_SUBMIT,
        profiling_command_start = CL_PROFILING_COMMAND_START,
        profiling_command_end = CL_PROFILING_COMMAND_END
    };

    /// Creates a null event object.
    event()
        : m_event(0)
    {
    }

    explicit event(cl_event event, bool retain = true)
        : m_event(event)
    {
        if(m_event && retain){
            clRetainEvent(event);
        }
    }

    event(const event &other)
        : m_event(other.m_event)
    {
        if(m_event){
            clRetainEvent(m_event);
        }
    }

    event(BOOST_RV_REF(event) other)
        : m_event(other.m_event)
    {
        other.m_event = 0;
    }

    event& operator=(const event &other)
    {
        if(this != &other){
            if(m_event){
                clReleaseEvent(m_event);
            }

            m_event = other.m_event;

            if(m_event){
                clRetainEvent(m_event);
            }
        }

        return *this;
    }

    event& operator=(BOOST_RV_REF(event) other)
    {
        if(this != &other){
            if(m_event){
                clReleaseEvent(m_event);
            }

            m_event = other.m_event;
            other.m_event = 0;
        }

        return *this;
    }

    /// Destroys the event object.
    ~event()
    {
        if(m_event){
            BOOST_COMPUTE_ASSERT_CL_SUCCESS(
                clReleaseEvent(m_event)
            );
        }
    }

    /// Returns a reference to the underlying OpenCL event object.
    cl_event& get() const
    {
        return const_cast<cl_event &>(m_event);
    }

    /// Returns the status of the event.
    cl_int get_status() const
    {
        return get_info<cl_int>(CL_EVENT_COMMAND_EXECUTION_STATUS);
    }

    /// Returns the command type for the event.
    cl_command_type get_command_type() const
    {
        return get_info<cl_command_type>(CL_EVENT_COMMAND_TYPE);
    }

    /// Returns information about the event.
    ///
    /// \see_opencl_ref{clGetEventInfo}
    template<class T>
    T get_info(cl_event_info info) const
    {
        return detail::get_object_info<T>(clGetEventInfo, m_event, info);
    }

    /// Returns profiling information for the event.
    ///
    /// \see_opencl_ref{clGetEventProfilingInfo}
    template<class T>
    T get_profiling_info(cl_profiling_info info) const
    {
        return detail::get_object_info<T>(clGetEventProfilingInfo,
                                          m_event,
                                          info);
    }

    /// Blocks until the actions corresponding to the event have
    /// completed.
    void wait()
    {
        cl_int ret = clWaitForEvents(1, &m_event);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }
    }

    #if defined(CL_VERSION_1_1) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Registers a function to be called when the event status changes to
    /// \p status (by default CL_COMPLETE). The callback is passed the OpenCL
    /// event object, the event status, and a pointer to arbitrary user data.
    ///
    /// \see_opencl_ref{clSetEventCallback}
    ///
    /// \opencl_version_warning{1,1}
    void set_callback(void (BOOST_COMPUTE_CL_CALLBACK *callback)(
                          cl_event event, cl_int status, void *user_data
                      ),
                      cl_int status = CL_COMPLETE,
                      void *user_data = 0)
    {
        cl_int ret = clSetEventCallback(m_event, status, callback, user_data);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }
    }
    #endif // CL_VERSION_1_1

    /// Returns \c true if the event is the same as \p other.
    bool operator==(const event &other) const
    {
        return m_event == other.m_event;
    }

    /// Returns \c true if the event is different from \p other.
    bool operator!=(const event &other) const
    {
        return m_event != other.m_event;
    }

    /// \internal_
    operator cl_event() const
    {
        return m_event;
    }

protected:
    cl_event m_event;

private:
    BOOST_COPYABLE_AND_MOVABLE(event)
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_EVENT_HPP
