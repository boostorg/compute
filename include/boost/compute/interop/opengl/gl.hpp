//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_INTEROP_OPENGL_GL_HPP
#define BOOST_COMPUTE_INTEROP_OPENGL_GL_HPP

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#if defined(_WIN32)
// Avoid error: 'APIENTRY' : illegal use of type 'void'
#include "windows.h"
#endif
#include <GL/gl.h>
#endif

#endif // BOOST_COMPUTE_INTEROP_OPENGL_GL_HPP
