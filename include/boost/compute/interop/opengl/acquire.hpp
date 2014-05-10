//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_INTEROP_OPENGL_ACQUIRE_HPP
#define BOOST_COMPUTE_INTEROP_OPENGL_ACQUIRE_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/interop/opengl/cl_gl.hpp>
#include <boost/compute/interop/opengl/opengl_buffer.hpp>

namespace boost {
namespace compute {

/// Enqueues a command to acquire the specified OpenGL memory objects.
///
/// \see_opencl_ref{clEnqueueAcquireGLObjects}
inline void opengl_enqueue_acquire_gl_objects(size_t num_objects,
                                              const cl_mem *mem_objects,
                                              command_queue &queue)
{
    cl_int ret = clEnqueueAcquireGLObjects(queue.get(),
                                           num_objects,
                                           mem_objects,
                                           0,
                                           0,
                                           0);
    if(ret != CL_SUCCESS){
        BOOST_THROW_EXCEPTION(opencl_error(ret));
    }
}

/// Enqueues a command to release the specified OpenGL memory objects.
///
/// \see_opencl_ref{clEnqueueReleaseGLObjects}
inline void opengl_enqueue_release_gl_objects(size_t num_objects,
                                              const cl_mem *mem_objects,
                                              command_queue &queue)
{
    cl_int ret = clEnqueueReleaseGLObjects(queue.get(),
                                           num_objects,
                                           mem_objects,
                                           0,
                                           0,
                                           0);
    if(ret != CL_SUCCESS){
        BOOST_THROW_EXCEPTION(opencl_error(ret));
    }
}

/// Enqueues a command to acquire the specified OpenGL buffer.
///
/// \see_opencl_ref{clEnqueueAcquireGLObjects}
inline void opengl_enqueue_acquire_buffer(const opengl_buffer &buffer,
                                          command_queue &queue)
{
    BOOST_ASSERT(buffer.get_context() == queue.get_context());

    opengl_enqueue_acquire_gl_objects(1, &buffer.get(), queue);
}

/// Enqueues a command to release the specified OpenGL buffer.
///
/// \see_opencl_ref{clEnqueueReleaseGLObjects}
inline void opengl_enqueue_release_buffer(const opengl_buffer &buffer,
                                          command_queue &queue)
{
    BOOST_ASSERT(buffer.get_context() == queue.get_context());

    opengl_enqueue_release_gl_objects(1, &buffer.get(), queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_INTEROP_OPENGL_ACQUIRE_HPP
