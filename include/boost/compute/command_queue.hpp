//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_COMMAND_QUEUE_H
#define BOOST_COMPUTE_COMMAND_QUEUE_H

#include <cstddef>

#include <boost/assert.hpp>
#include <boost/move/move.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/event.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/image2d.hpp>
#include <boost/compute/image3d.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/wait_list.hpp>
#include <boost/compute/detail/get_object_info.hpp>
#include <boost/compute/detail/assert_cl_success.hpp>

namespace boost {
namespace compute {
namespace detail {

inline void BOOST_COMPUTE_CL_CALLBACK
nullary_native_kernel_trampoline(void *user_func_ptr)
{
    void (*user_func)();
    std::memcpy(&user_func, user_func_ptr, sizeof(user_func));
    user_func();
}

} // end detail namespace

/// \class command_queue
/// \brief A command queue.
///
/// Command queues provide the interface for interacting with compute
/// devices. The command_queue class provides methods to copy data to
/// and from a compute device as well as execute compute kernels.
///
/// Command queues are created for a compute device within a compute
/// context.
///
/// The default command queue for the system can be obtained with the
/// system::default_queue() method.
class command_queue
{
public:
    enum properties {
        enable_profiling = CL_QUEUE_PROFILING_ENABLE,
        enable_out_of_order_execution = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    };

    enum map_flags {
        map_read = CL_MAP_READ,
        map_write = CL_MAP_WRITE
    };

    /// Creates a null command queue.
    command_queue()
        : m_queue(0)
    {
    }

    explicit command_queue(cl_command_queue queue, bool retain = true)
        : m_queue(queue)
    {
        if(m_queue && retain){
            clRetainCommandQueue(m_queue);
        }
    }

    /// Creates a command queue in \p context for \p device with
    /// \p properties.
    ///
    /// \see_opencl_ref{clCreateCommandQueue}
    command_queue(const context &context,
                  const device &device,
                  cl_command_queue_properties properties = 0)
    {
        BOOST_ASSERT(device.id() != 0);

        cl_int error = 0;
        m_queue = clCreateCommandQueue(context,
                                       device.id(),
                                       properties,
                                       &error);
        if(!m_queue){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }
    }

    /// Creates a new command queue object as a copy of \p other.
    command_queue(const command_queue &other)
        : m_queue(other.m_queue)
    {
        if(m_queue){
            clRetainCommandQueue(m_queue);
        }
    }

    command_queue(BOOST_RV_REF(command_queue) other)
        : m_queue(other.m_queue)
    {
        other.m_queue = 0;
    }

    command_queue& operator=(const command_queue &other)
    {
        if(this != &other){
            if(m_queue){
                clReleaseCommandQueue(m_queue);
            }

            m_queue = other.m_queue;

            if(m_queue){
                clRetainCommandQueue(m_queue);
            }
        }

        return *this;
    }

    command_queue& operator=(BOOST_RV_REF(command_queue) other)
    {
        if(this != &other){
            if(m_queue){
                clReleaseCommandQueue(m_queue);
            }

            m_queue = other.m_queue;
            other.m_queue = 0;
        }

        return *this;
    }

    /// Destroys the command queue.
    ///
    /// \see_opencl_ref{clReleaseCommandQueue}
    ~command_queue()
    {
        if(m_queue){
            // finsh any outstanding operations before destoying the queue
            finish();

            // release the memory for the command queue
            BOOST_COMPUTE_ASSERT_CL_SUCCESS(
                clReleaseCommandQueue(m_queue)
            );
        }
    }

    /// Returns the underlying OpenCL command queue.
    cl_command_queue& get() const
    {
        return const_cast<cl_command_queue &>(m_queue);
    }

    /// Returns the device that the command queue issues commands to.
    device get_device() const
    {
        return device(get_info<cl_device_id>(CL_QUEUE_DEVICE));
    }

    /// Returns the context for the command queue.
    context get_context() const
    {
        return context(get_info<cl_context>(CL_QUEUE_CONTEXT));
    }

    /// Returns information about the command queue.
    ///
    /// \see_opencl_ref{clGetCommandQueueInfo}
    template<class T>
    T get_info(cl_command_queue_info info) const
    {
        return detail::get_object_info<T>(clGetCommandQueueInfo, m_queue, info);
    }

    /// Returns the properties for the command queue.
    cl_command_queue_properties get_properties() const
    {
        return get_info<cl_command_queue_properties>(CL_QUEUE_PROPERTIES);
    }

    /// \internal_
    cl_int enqueue_read_buffer(const buffer &buffer, void *host_ptr)
    {
        return enqueue_read_buffer(buffer, 0, buffer.size(), host_ptr);
    }

    /// \internal_
    cl_int enqueue_read_buffer(const buffer &buffer,
                               size_t size,
                               void *host_ptr)
    {
        return enqueue_read_buffer(buffer, 0, size, host_ptr);
    }

    /// Enqueues a command to read data from \p buffer to host memory.
    ///
    /// \see_opencl_ref{clEnqueueReadBuffer}
    ///
    /// \see copy()
    cl_int enqueue_read_buffer(const buffer &buffer,
                               size_t offset,
                               size_t size,
                               void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(size <= buffer.size());
        BOOST_ASSERT(buffer.get_context() == this->get_context());
        BOOST_ASSERT(host_ptr != 0);

        cl_int ret = clEnqueueReadBuffer(m_queue,
                                         buffer.get(),
                                         true,
                                         offset,
                                         size,
                                         host_ptr,
                                         0,
                                         0,
                                         0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to read data from \p buffer to host memory. The
    /// copy will be performed asynchronously.
    ///
    /// \see_opencl_ref{clEnqueueReadBuffer}
    ///
    /// \see copy_async()
    event enqueue_read_buffer_async(const buffer &buffer,
                                    size_t offset,
                                    size_t size,
                                    void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(size <= buffer.size());
        BOOST_ASSERT(buffer.get_context() == this->get_context());
        BOOST_ASSERT(host_ptr != 0);

        event event_;

        cl_int ret = clEnqueueReadBuffer(m_queue,
                                         buffer.get(),
                                         true,
                                         offset,
                                         size,
                                         host_ptr,
                                         0,
                                         0,
                                         &event_.get());
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }

    #if defined(CL_VERSION_1_1) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Enqueues a command to read a rectangular region from \p buffer to
    /// host memory.
    ///
    /// \see_opencl_ref{clEnqueueReadBufferRect}
    ///
    /// \opencl_version_warning{1,1}
    cl_int enqueue_read_buffer_rect(const buffer &buffer,
                                    const size_t buffer_origin[3],
                                    const size_t host_origin[3],
                                    const size_t region[3],
                                    size_t buffer_row_pitch,
                                    size_t buffer_slice_pitch,
                                    size_t host_row_pitch,
                                    size_t host_slice_pitch,
                                    void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(buffer.get_context() == this->get_context());
        BOOST_ASSERT(host_ptr != 0);

        cl_int ret = clEnqueueReadBufferRect(m_queue,
                                             buffer.get(),
                                             CL_TRUE,
                                             buffer_origin,
                                             host_origin,
                                             region,
                                             buffer_row_pitch,
                                             buffer_slice_pitch,
                                             host_row_pitch,
                                             host_slice_pitch,
                                             host_ptr,
                                             0,
                                             0,
                                             0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }
    #endif // CL_VERSION_1_1

    /// \internal_
    cl_int enqueue_write_buffer(const buffer &buffer, const void *host_ptr)
    {
        return enqueue_write_buffer(buffer, 0, buffer.size(), host_ptr);
    }

    /// \internal_
    cl_int enqueue_write_buffer(const buffer &buffer,
                                size_t size,
                                const void *host_ptr)
    {
        return enqueue_write_buffer(buffer, 0, size, host_ptr);
    }

    /// Enqueues a command to write data from host memory to \p buffer.
    ///
    /// \see_opencl_ref{clEnqueueWriteBuffer}
    ///
    /// \see copy()
    cl_int enqueue_write_buffer(const buffer &buffer,
                                size_t offset,
                                size_t size,
                                const void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(size <= buffer.size());
        BOOST_ASSERT(buffer.get_context() == this->get_context());
        BOOST_ASSERT(host_ptr != 0);

        cl_int ret = clEnqueueWriteBuffer(m_queue,
                                          buffer.get(),
                                          CL_TRUE,
                                          offset,
                                          size,
                                          host_ptr,
                                          0,
                                          0,
                                          0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to write data from host memory to \p buffer.
    /// The copy is performed asynchronously.
    ///
    /// \see_opencl_ref{clEnqueueWriteBuffer}
    ///
    /// \see copy_async()
    event enqueue_write_buffer_async(const buffer &buffer,
                                     size_t offset,
                                     size_t size,
                                     const void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(size <= buffer.size());
        BOOST_ASSERT(buffer.get_context() == this->get_context());
        BOOST_ASSERT(host_ptr != 0);

        event event_;

        cl_int ret = clEnqueueWriteBuffer(m_queue,
                                          buffer.get(),
                                          CL_FALSE,
                                          offset,
                                          size,
                                          host_ptr,
                                          0,
                                          0,
                                          &event_.get());
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }

    #if defined(CL_VERSION_1_1) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Enqueues a command to write a rectangular region from host memory
    /// to \p buffer.
    ///
    /// \see_opencl_ref{clEnqueueWriteBufferRect}
    ///
    /// \opencl_version_warning{1,1}
    cl_int enqueue_write_buffer_rect(const buffer &buffer,
                                     const size_t buffer_origin[3],
                                     const size_t host_origin[3],
                                     const size_t region[3],
                                     size_t buffer_row_pitch,
                                     size_t buffer_slice_pitch,
                                     size_t host_row_pitch,
                                     size_t host_slice_pitch,
                                     void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(buffer.get_context() == this->get_context());
        BOOST_ASSERT(host_ptr != 0);

        cl_int ret = clEnqueueWriteBufferRect(m_queue,
                                              buffer.get(),
                                              CL_TRUE,
                                              buffer_origin,
                                              host_origin,
                                              region,
                                              buffer_row_pitch,
                                              buffer_slice_pitch,
                                              host_row_pitch,
                                              host_slice_pitch,
                                              host_ptr,
                                              0,
                                              0,
                                              0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }
    #endif // CL_VERSION_1_1

    /// Enqueues a command to copy data from \p src_buffer to
    /// \p dst_buffer.
    ///
    /// \see_opencl_ref{clEnqueueCopyBuffer}
    ///
    /// \see copy()
    event enqueue_copy_buffer(const buffer &src_buffer,
                              const buffer &dst_buffer,
                              size_t src_offset,
                              size_t dst_offset,
                              size_t size)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_offset + size <= src_buffer.size());
        BOOST_ASSERT(dst_offset + size <= dst_buffer.size());
        BOOST_ASSERT(src_buffer.get_context() == this->get_context());
        BOOST_ASSERT(dst_buffer.get_context() == this->get_context());

        event event_;

        cl_int ret = clEnqueueCopyBuffer(m_queue,
                                         src_buffer.get(),
                                         dst_buffer.get(),
                                         src_offset,
                                         dst_offset,
                                         size,
                                         0,
                                         0,
                                         &event_.get());
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }

    #if defined(CL_VERSION_1_1) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Enqueues a command to copy a rectangular region from
    /// \p src_buffer to \p dst_buffer.
    ///
    /// \see_opencl_ref{clEnqueueCopyBufferRect}
    ///
    /// \opencl_version_warning{1,1}
    cl_int enqueue_copy_buffer_rect(const buffer &src_buffer,
                                    const buffer &dst_buffer,
                                    const size_t src_origin[3],
                                    const size_t dst_origin[3],
                                    const size_t region[3],
                                    size_t buffer_row_pitch,
                                    size_t buffer_slice_pitch,
                                    size_t host_row_pitch,
                                    size_t host_slice_pitch)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_buffer.get_context() == this->get_context());
        BOOST_ASSERT(dst_buffer.get_context() == this->get_context());

        cl_int ret = clEnqueueCopyBufferRect(m_queue,
                                             src_buffer.get(),
                                             dst_buffer.get(),
                                             src_origin,
                                             dst_origin,
                                             region,
                                             buffer_row_pitch,
                                             buffer_slice_pitch,
                                             host_row_pitch,
                                             host_slice_pitch,
                                             0,
                                             0,
                                             0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }
    #endif // CL_VERSION_1_1

    #if defined(CL_VERSION_1_2) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Enqueues a command to fill \p buffer with \p pattern.
    ///
    /// \see_opencl_ref{clEnqueueFillBuffer}
    ///
    /// \opencl_version_warning{1,2}
    ///
    /// \see fill()
    event enqueue_fill_buffer(const buffer &buffer,
                              const void *pattern,
                              size_t pattern_size,
                              size_t offset,
                              size_t size)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(offset + size <= buffer.size());
        BOOST_ASSERT(buffer.get_context() == this->get_context());

        event event_;

        cl_int ret = clEnqueueFillBuffer(m_queue,
                                         buffer.get(),
                                         pattern,
                                         pattern_size,
                                         offset,
                                         size,
                                         0,
                                         0,
                                         &event_.get());
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }
    #endif // CL_VERSION_1_2

    /// Enqueues a command to map \p buffer into the host address space.
    ///
    /// \see_opencl_ref{clEnqueueMapBuffer}
    void* enqueue_map_buffer(const buffer &buffer,
                             cl_map_flags flags,
                             size_t offset,
                             size_t size)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(offset + size <= buffer.size());
        BOOST_ASSERT(buffer.get_context() == this->get_context());

        cl_int ret = 0;
        void *pointer = clEnqueueMapBuffer(m_queue,
                                           buffer.get(),
                                           CL_TRUE,
                                           flags,
                                           offset,
                                           size,
                                           0,
                                           0,
                                           0,
                                           &ret);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return pointer;
    }

    /// Enqueues a command to unmap \p buffer from the host memory space.
    ///
    /// \see_opencl_ref{clEnqueueUnmapMemObject}
    cl_int enqueue_unmap_buffer(const buffer &buffer, void *mapped_ptr)
    {
        BOOST_ASSERT(buffer.get_context() == this->get_context());

        return enqueue_unmap_mem_object(buffer.get(), mapped_ptr);
    }

    /// Enqueues a command to unmap \p mem from the host memory space.
    ///
    /// \see_opencl_ref{clEnqueueUnmapMemObject}
    cl_int enqueue_unmap_mem_object(cl_mem mem, void *mapped_ptr)
    {
        BOOST_ASSERT(m_queue != 0);

        cl_int ret = clEnqueueUnmapMemObject(m_queue,
                                             mem,
                                             mapped_ptr,
                                             0,
                                             0,
                                             0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to read data from \p image to host memory.
    ///
    /// \see_opencl_ref{clEnqueueReadImage}
    cl_int enqueue_read_image(const image2d &image,
                              const size_t origin[2],
                              const size_t region[2],
                              size_t row_pitch,
                              void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(image.get_context() == this->get_context());

        const size_t origin3[3] = { origin[0], origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueReadImage(m_queue,
                                        image.get(),
                                        CL_TRUE,
                                        origin3,
                                        region3,
                                        row_pitch,
                                        0,
                                        host_ptr,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to read data from \p image to host memory.
    ///
    /// \see_opencl_ref{clEnqueueReadImage}
    cl_int enqueue_read_image(const image3d &image,
                              const size_t origin[3],
                              const size_t region[3],
                              size_t row_pitch,
                              size_t slice_pitch,
                              void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(image.get_context() == this->get_context());

        cl_int ret = clEnqueueReadImage(m_queue,
                                        image.get(),
                                        CL_TRUE,
                                        origin,
                                        region,
                                        row_pitch,
                                        slice_pitch,
                                        host_ptr,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to write data from host memory to \p image.
    ///
    /// \see_opencl_ref{clEnqueueWriteImage}
    cl_int enqueue_write_image(const image2d &image,
                               const size_t origin[2],
                               const size_t region[2],
                               size_t input_row_pitch,
                               const void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(image.get_context() == this->get_context());

        const size_t origin3[3] = { origin[0], origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueWriteImage(m_queue,
                                         image.get(),
                                         CL_TRUE,
                                         origin3,
                                         region3,
                                         input_row_pitch,
                                         0,
                                         host_ptr,
                                         0,
                                         0,
                                         0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to write data from host memory to \p image.
    ///
    /// \see_opencl_ref{clEnqueueWriteImage}
    cl_int enqueue_write_image(const image3d &image,
                               const size_t origin[3],
                               const size_t region[3],
                               size_t input_row_pitch,
                               size_t input_slice_pitch,
                               const void *host_ptr)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(image.get_context() == this->get_context());

        cl_int ret = clEnqueueWriteImage(m_queue,
                                         image.get(),
                                         CL_TRUE,
                                         origin,
                                         region,
                                         input_row_pitch,
                                         input_slice_pitch,
                                         host_ptr,
                                         0,
                                         0,
                                         0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_image to \p dst_image.
    ///
    /// \see_opencl_ref{clEnqueueCopyImage}
    cl_int enqueue_copy_image(const image2d &src_image,
                              const image2d &dst_image,
                              const size_t src_origin[2],
                              const size_t dst_origin[2],
                              const size_t region[2])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_image.get_context() == this->get_context());
        BOOST_ASSERT(dst_image.get_context() == this->get_context());
        BOOST_ASSERT_MSG(src_image.get_format() == dst_image.get_format(),
                         "Source and destination image formats must match.");

        const size_t src_origin3[3] = { src_origin[0], src_origin[1], size_t(0) };
        const size_t dst_origin3[3] = { dst_origin[0], dst_origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueCopyImage(m_queue,
                                        src_image.get(),
                                        dst_image.get(),
                                        src_origin3,
                                        dst_origin3,
                                        region3,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_image to \p dst_image.
    ///
    /// \see_opencl_ref{clEnqueueCopyImage}
    cl_int enqueue_copy_image(const image2d &src_image,
                              const image3d &dst_image,
                              const size_t src_origin[2],
                              const size_t dst_origin[3],
                              const size_t region[2])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_image.get_context() == this->get_context());
        BOOST_ASSERT(dst_image.get_context() == this->get_context());
        BOOST_ASSERT_MSG(src_image.get_format() == dst_image.get_format(),
                         "Source and destination image formats must match.");

        const size_t src_origin3[3] = { src_origin[0], src_origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueCopyImage(m_queue,
                                        src_image.get(),
                                        dst_image.get(),
                                        src_origin3,
                                        dst_origin,
                                        region3,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_image to \p dst_image.
    ///
    /// \see_opencl_ref{clEnqueueCopyImage}
    cl_int enqueue_copy_image(const image3d &src_image,
                              const image2d &dst_image,
                              const size_t src_origin[3],
                              const size_t dst_origin[2],
                              const size_t region[2])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_image.get_context() == this->get_context());
        BOOST_ASSERT(dst_image.get_context() == this->get_context());
        BOOST_ASSERT_MSG(src_image.get_format() == dst_image.get_format(),
                         "Source and destination image formats must match.");

        const size_t dst_origin3[3] = { dst_origin[0], dst_origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueCopyImage(m_queue,
                                        src_image.get(),
                                        dst_image.get(),
                                        src_origin,
                                        dst_origin3,
                                        region3,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_image to \p dst_image.
    ///
    /// \see_opencl_ref{clEnqueueCopyImage}
    cl_int enqueue_copy_image(const image3d &src_image,
                              const image3d &dst_image,
                              const size_t src_origin[3],
                              const size_t dst_origin[3],
                              const size_t region[3])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_image.get_context() == this->get_context());
        BOOST_ASSERT(dst_image.get_context() == this->get_context());
        BOOST_ASSERT_MSG(src_image.get_format() == dst_image.get_format(),
                         "Source and destination image formats must match.");

        cl_int ret = clEnqueueCopyImage(m_queue,
                                        src_image.get(),
                                        dst_image.get(),
                                        src_origin,
                                        dst_origin,
                                        region,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_image to \p dst_buffer.
    ///
    /// \see_opencl_ref{clEnqueueCopyImageToBuffer}
    cl_int enqueue_copy_image_to_buffer(const image2d &src_image,
                                        const buffer &dst_buffer,
                                        const size_t src_origin[2],
                                        const size_t region[2],
                                        size_t dst_offset)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_image.get_context() == this->get_context());
        BOOST_ASSERT(dst_buffer.get_context() == this->get_context());

        const size_t src_origin3[3] = { src_origin[0], src_origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueCopyImageToBuffer(m_queue,
                                                src_image.get(),
                                                dst_buffer.get(),
                                                src_origin3,
                                                region3,
                                                dst_offset,
                                                0,
                                                0,
                                                0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_image to \p dst_buffer.
    ///
    /// \see_opencl_ref{clEnqueueCopyImageToBuffer}
    cl_int enqueue_copy_image_to_buffer(const image3d &src_image,
                                        const buffer &dst_buffer,
                                        const size_t src_origin[3],
                                        const size_t region[3],
                                        size_t dst_offset)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_image.get_context() == this->get_context());
        BOOST_ASSERT(dst_buffer.get_context() == this->get_context());

        cl_int ret = clEnqueueCopyImageToBuffer(m_queue,
                                                src_image.get(),
                                                dst_buffer.get(),
                                                src_origin,
                                                region,
                                                dst_offset,
                                                0,
                                                0,
                                                0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_buffer to \p dst_image.
    ///
    /// \see_opencl_ref{clEnqueueCopyBufferToImage}
    cl_int enqueue_copy_buffer_to_image(const buffer &src_buffer,
                                        const image2d &dst_image,
                                        size_t src_offset,
                                        const size_t dst_origin[3],
                                        const size_t region[3])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_buffer.get_context() == this->get_context());
        BOOST_ASSERT(dst_image.get_context() == this->get_context());

        const size_t dst_origin3[3] = { dst_origin[0], dst_origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueCopyBufferToImage(m_queue,
                                                src_buffer.get(),
                                                dst_image.get(),
                                                src_offset,
                                                dst_origin3,
                                                region3,
                                                0,
                                                0,
                                                0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to copy data from \p src_buffer to \p dst_image.
    ///
    /// \see_opencl_ref{clEnqueueCopyBufferToImage}
    cl_int enqueue_copy_buffer_to_image(const buffer &src_buffer,
                                        const image3d &dst_image,
                                        size_t src_offset,
                                        const size_t dst_origin[3],
                                        const size_t region[3])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(src_buffer.get_context() == this->get_context());
        BOOST_ASSERT(dst_image.get_context() == this->get_context());

        cl_int ret = clEnqueueCopyBufferToImage(m_queue,
                                                src_buffer.get(),
                                                dst_image.get(),
                                                src_offset,
                                                dst_origin,
                                                region,
                                                0,
                                                0,
                                                0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    #if defined(CL_VERSION_1_2) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Enqueues a command to fill \p image with \p fill_color.
    ///
    /// \see_opencl_ref{clEnqueueFillImage}
    ///
    /// \opencl_version_warning{1,2}
    cl_int enqueue_fill_image(const image2d &image,
                              const void *fill_color,
                              const size_t origin[2],
                              const size_t region[2])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(image.get_context() == this->get_context());

        const size_t origin3[3] = { origin[0], origin[1], size_t(0) };
        const size_t region3[3] = { region[0], region[1], size_t(1) };

        cl_int ret = clEnqueueFillImage(m_queue,
                                        image.get(),
                                        fill_color,
                                        origin3,
                                        region3,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to fill \p image with \p fill_color.
    ///
    /// \see_opencl_ref{clEnqueueFillImage}
    ///
    /// \opencl_version_warning{1,2}
    cl_int enqueue_fill_image(const image3d &image,
                              const void *fill_color,
                              const size_t origin[3],
                              const size_t region[3])
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(image.get_context() == this->get_context());

        cl_int ret = clEnqueueFillImage(m_queue,
                                        image.get(),
                                        fill_color,
                                        origin,
                                        region,
                                        0,
                                        0,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    /// Enqueues a command to migrate \p mem_objects.
    ///
    /// \see_opencl_ref{clEnqueueMigrateMemObjects}
    ///
    /// \opencl_version_warning{1,2}
    void enqueue_migrate_memory_objects(uint_ num_mem_objects,
                                        const cl_mem *mem_objects,
                                        cl_mem_migration_flags flags)
    {
        BOOST_ASSERT(m_queue != 0);

        cl_int ret = clEnqueueMigrateMemObjects(m_queue,
                                                num_mem_objects,
                                                mem_objects,
                                                flags,
                                                0,
                                                0,
                                                0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }
    }
    #endif // CL_VERSION_1_2

    /// Enqueues a kernel for execution.
    ///
    /// \see_opencl_ref{clEnqueueNDRangeKernel}
    event enqueue_nd_range_kernel(const kernel &kernel,
                                  size_t work_dim,
                                  const size_t *global_work_offset,
                                  const size_t *global_work_size,
                                  const size_t *local_work_size = 0)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(kernel.get_context() == this->get_context());
        BOOST_ASSERT(work_dim > 0 && work_dim <= CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);

        event event_;

        cl_int ret = clEnqueueNDRangeKernel(m_queue,
                                            kernel,
                                            static_cast<cl_uint>(work_dim),
                                            global_work_offset,
                                            global_work_size,
                                            local_work_size ? local_work_size : 0,
                                            0,
                                            0,
                                            &event_.get());
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }

    /// \internal_
    event enqueue_1d_range_kernel(const kernel &kernel,
                                  size_t global_work_offset,
                                  size_t global_work_size)
    {
        return enqueue_nd_range_kernel(kernel,
                                       1,
                                       &global_work_offset,
                                       &global_work_size,
                                       0);
    }

    /// \internal_
    event enqueue_1d_range_kernel(const kernel &kernel,
                                  size_t global_work_offset,
                                  size_t global_work_size,
                                  size_t local_work_size)
    {
        return enqueue_nd_range_kernel(kernel,
                                       1,
                                       &global_work_offset,
                                       &global_work_size,
                                       local_work_size ? &local_work_size : 0);
    }

    /// Enqueues a kernel to execute using a single work-item.
    ///
    /// \see_opencl_ref{clEnqueueTask}
    event enqueue_task(const kernel &kernel)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(kernel.get_context() == this->get_context());

        event event_;

        cl_int ret = clEnqueueTask(m_queue, kernel, 0, 0, &event_.get());
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }

    /// Enqueues a function to execute on the host.
    event enqueue_native_kernel(void (BOOST_COMPUTE_CL_CALLBACK *user_func)(void *),
                                void *args,
                                size_t cb_args,
                                uint_ num_mem_objects,
                                const cl_mem *mem_list,
                                const void **args_mem_loc)
    {
        BOOST_ASSERT(m_queue != 0);

        event event_;
        cl_int ret = clEnqueueNativeKernel(
            m_queue,
            user_func,
            args,
            cb_args,
            num_mem_objects,
            mem_list,
            args_mem_loc,
            0,
            0,
            &event_.get()
        );
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }

    /// Convience overload for enqueue_native_kernel() which enqueues a
    /// native kernel on the host with a nullary function.
    event enqueue_native_kernel(void (BOOST_COMPUTE_CL_CALLBACK *user_func)(void))
    {
        return enqueue_native_kernel(
            detail::nullary_native_kernel_trampoline,
            reinterpret_cast<void *>(&user_func),
            sizeof(user_func),
            0,
            0,
            0
        );
    }

    /// Flushes the command queue.
    ///
    /// \see_opencl_ref{clFlush}
    void flush()
    {
        BOOST_ASSERT(m_queue != 0);

        clFlush(m_queue);
    }

    /// Blocks until all outstanding commands in the queue have finished.
    ///
    /// \see_opencl_ref{clFinish}
    void finish()
    {
        BOOST_ASSERT(m_queue != 0);

        clFinish(m_queue);
    }

    /// Enqueues a barrier in the queue.
    void enqueue_barrier()
    {
        BOOST_ASSERT(m_queue != 0);

        #ifdef CL_VERSION_1_2
        clEnqueueBarrierWithWaitList(m_queue, 0, 0, 0);
        #else
        clEnqueueBarrier(m_queue);
        #endif
    }

    #if defined(CL_VERSION_1_2) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Enqueues a barrier in the queue after \p events.
    ///
    /// \opencl_version_warning{1,2}
    void enqueue_barrier(const wait_list &events)
    {
        BOOST_ASSERT(m_queue != 0);

        clEnqueueBarrierWithWaitList(
            m_queue, events.size(), events.get_event_ptr(), 0
        );
    }
    #endif // CL_VERSION_1_2

    /// Enqueues a marker in the queue and returns an event that can be
    /// used to track its progress.
    event enqueue_marker()
    {
        event event_;

        #ifdef CL_VERSION_1_2
        cl_int ret = clEnqueueMarkerWithWaitList(m_queue, 0, 0, &event_.get());
        #else
        cl_int ret = clEnqueueMarker(m_queue, &event_.get());
        #endif

        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }

    #if defined(CL_VERSION_1_2) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Enqueues a marker after \p events in the queue and returns an
    /// event that can be used to track its progress.
    ///
    /// \opencl_version_warning{1,2}
    event enqueue_marker(const wait_list &events)
    {
        event event_;

        cl_int ret = clEnqueueMarkerWithWaitList(
            m_queue, events.size(), events.get_event_ptr(), &event_.get()
        );

        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return event_;
    }
    #endif // CL_VERSION_1_2

    /// \internal_
    operator cl_command_queue() const
    {
        return m_queue;
    }

private:
    BOOST_COPYABLE_AND_MOVABLE(command_queue)

    cl_command_queue m_queue;
};

inline buffer buffer::clone(command_queue &queue) const
{
    buffer copy(this->get_context(), this->size());
    queue.enqueue_copy_buffer(*this, copy, 0, 0, this->size());
    return copy;
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_COMMAND_QUEUE_H
