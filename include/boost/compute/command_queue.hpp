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

#include <boost/compute/cl.hpp>
#include <boost/compute/event.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/image2d.hpp>
#include <boost/compute/image3d.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/detail/get_object_info.hpp>

#ifdef BOOST_COMPUTE_HAVE_GL
#include <boost/compute/cl_gl.hpp>
#endif

namespace boost {
namespace compute {

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

    command_queue(const command_queue &other)
        : m_queue(other.m_queue)
    {
        if(m_queue){
            clRetainCommandQueue(m_queue);
        }
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

    ~command_queue()
    {
        if(m_queue){
            // finsh any outstanding operations before destoying the queue
            finish();

            // release the memory for the command queue
            clReleaseCommandQueue(m_queue);
        }
    }

    cl_command_queue& get() const
    {
        return const_cast<cl_command_queue &>(m_queue);
    }

    device get_device() const
    {
        return device(get_info<cl_device_id>(CL_QUEUE_DEVICE));
    }

    context get_context() const
    {
        return context(get_info<cl_context>(CL_QUEUE_CONTEXT));
    }

    template<class T>
    T get_info(cl_command_queue_info info) const
    {
        return detail::get_object_info<T>(clGetCommandQueueInfo, m_queue, info);
    }

    cl_command_queue_properties get_properties() const
    {
        return get_info<cl_command_queue_properties>(CL_QUEUE_PROPERTIES);
    }

    cl_int enqueue_read_buffer(const buffer &buffer, void *host_ptr)
    {
        return enqueue_read_buffer(buffer, 0, buffer.size(), host_ptr);
    }

    cl_int enqueue_read_buffer(const buffer &buffer,
                               size_t size,
                               void *host_ptr)
    {
        return enqueue_read_buffer(buffer, 0, size, host_ptr);
    }

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

    #ifdef CL_VERSION_1_1
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

    cl_int enqueue_write_buffer(const buffer &buffer, const void *host_ptr)
    {
        return enqueue_write_buffer(buffer, 0, buffer.size(), host_ptr);
    }

    cl_int enqueue_write_buffer(const buffer &buffer,
                                size_t size,
                                const void *host_ptr)
    {
        return enqueue_write_buffer(buffer, 0, size, host_ptr);
    }

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

    #ifdef CL_VERSION_1_1
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

    cl_int enqueue_copy_buffer(const buffer &src_buffer,
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

        cl_int ret = clEnqueueCopyBuffer(m_queue,
                                         src_buffer.get(),
                                         dst_buffer.get(),
                                         src_offset,
                                         dst_offset,
                                         size,
                                         0,
                                         0,
                                         0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    #ifdef CL_VERSION_1_1
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

    #ifdef CL_VERSION_1_2
    cl_int enqueue_fill_buffer(const buffer &buffer,
                               const void *pattern,
                               size_t pattern_size,
                               size_t offset,
                               size_t size)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(offset + size <= buffer.size());
        BOOST_ASSERT(buffer.get_context() == this->get_context());

        cl_int ret = clEnqueueFillBuffer(m_queue,
                                         buffer.get(),
                                         pattern,
                                         pattern_size,
                                         offset,
                                         size,
                                         0,
                                         0,
                                         0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }
    #endif // CL_VERSION_1_2

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

    cl_int enqueue_unmap_buffer(const buffer &buffer, void *mapped_ptr)
    {
        BOOST_ASSERT(buffer.get_context() == this->get_context());

        return enqueue_unmap_mem_object(buffer.get(), mapped_ptr);
    }

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

    #ifdef CL_VERSION_1_2
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
    #endif // CL_VERSION_1_2

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

    cl_int enqueue_task(const kernel &kernel)
    {
        BOOST_ASSERT(m_queue != 0);
        BOOST_ASSERT(kernel.get_context() == this->get_context());

        cl_int ret = clEnqueueTask(m_queue, kernel, 0, 0, 0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return ret;
    }

    void flush()
    {
        BOOST_ASSERT(m_queue != 0);

        clFlush(m_queue);
    }

    void finish()
    {
        BOOST_ASSERT(m_queue != 0);

        clFinish(m_queue);
    }

    void enqueue_barrier()
    {
        BOOST_ASSERT(m_queue != 0);

        #ifdef CL_VERSION_1_2
        clEnqueueBarrierWithWaitList(m_queue, 0, 0, 0);
        #else
        clEnqueueBarrier(m_queue);
        #endif
    }

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

    operator cl_command_queue() const
    {
        return m_queue;
    }

    #ifdef BOOST_COMPUTE_HAVE_GL
    void enqueue_acquire_gl_objects(size_t num_objects,
                                    const cl_mem *mem_objects)
    {
        BOOST_ASSERT(m_queue != 0);

        cl_int ret = clEnqueueAcquireGLObjects(m_queue,
                                               num_objects,
                                               mem_objects,
                                               0,
                                               0,
                                               0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }
    }

    void enqueue_acquire_gl_buffer(const buffer &buffer)
    {
        BOOST_ASSERT(buffer.get_context() == this->get_context());

        enqueue_acquire_gl_objects(1, &buffer.get());
    }

    void enqueue_release_gl_objects(size_t num_objects,
                                    const cl_mem *mem_objects)
    {
        BOOST_ASSERT(m_queue != 0);

        cl_int ret = clEnqueueReleaseGLObjects(m_queue,
                                               num_objects,
                                               mem_objects,
                                               0,
                                               0,
                                               0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }
    }

    void enqueue_release_gl_buffer(const buffer &buffer)
    {
        BOOST_ASSERT(buffer.get_context() == this->get_context());

        enqueue_release_gl_objects(1, &buffer.get());
    }
    #endif // BOOST_COMPUTE_HAVE_GL

private:
    cl_command_queue m_queue;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_COMMAND_QUEUE_H
