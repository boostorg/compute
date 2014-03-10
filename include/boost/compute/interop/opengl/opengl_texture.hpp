//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_INTEROP_OPENGL_OPENGL_TEXTURE_HPP
#define BOOST_COMPUTE_INTEROP_OPENGL_OPENGL_TEXTURE_HPP

#include <boost/compute/memory_object.hpp>
#include <boost/compute/interop/opengl/gl.hpp>
#include <boost/compute/interop/opengl/cl_gl.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

/// \class opengl_texture
///
/// A OpenCL image2d for accessing an OpenGL texture object.
class opengl_texture : public memory_object
{
public:
    /// Creates a null OpenGL texture object.
    opengl_texture()
        : memory_object()
    {
    }

    /// Creates a new OpenGL texture object for \p mem.
    explicit opengl_texture(cl_mem mem, bool retain = true)
        : memory_object(mem, retain)
    {
    }

    /// Creates a new OpenGL texture object in \p context for \p texture
    /// with \p flags.
    ///
    /// \see_opencl_ref{clCreateFromGLTexture}
    opengl_texture(const context &context,
                   GLenum texture_target,
                   GLint miplevel,
                   GLuint texture,
                   cl_mem_flags flags = read_write)
    {
        cl_int error = 0;

        #ifdef CL_VERSION_1_2
        m_mem = clCreateFromGLTexture(context,
                                      flags,
                                      texture_target,
                                      miplevel,
                                      texture,
                                      &error);
        #else
        m_mem = clCreateFromGLTexture2D(context,
                                        flags,
                                        texture_target,
                                        miplevel,
                                        texture,
                                        &error);
        #endif

        if(!m_mem){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }
    }

    /// Creates a new OpenGL texture object as a copy of \p other.
    opengl_texture(const opengl_texture &other)
        : memory_object(other)
    {
    }

    /// Copies the OpenGL texture object from \p other.
    opengl_texture& operator=(const opengl_texture &other)
    {
        if(this != &other){
            memory_object::operator=(other);
        }

        return *this;
    }

    /// Destroys the texture object.
    ~opengl_texture()
    {
    }

    /// Returns information about the texture.
    ///
    /// \see_opencl_ref{clGetGLTextureInfo}
    template<class T>
    T get_texture_info(cl_gl_texture_info info) const
    {
        return detail::get_object_info<T>(clGetGLTextureInfo, m_mem, info);
    }
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_INTEROP_OPENGL_OPENGL_TEXTURE_HPP
