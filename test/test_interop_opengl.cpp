//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestInteropOpenGL
#include <boost/test/unit_test.hpp>

#include <boost/compute/interop/opengl.hpp>

BOOST_AUTO_TEST_CASE(opengl_buffer)
{
}

BOOST_AUTO_TEST_CASE(unsupported_extension_error)
{
    #ifdef __linux__
    #undef __linux__
    try {
        boost::compute::context context = boost::compute::opengl_create_shared_context();
    } catch(boost::compute::unsupported_extension_error& error) {
        BOOST_CHECK_EQUAL(std::string(error.what()), std::string("OpenCL extension cl_khr_gl_sharing not supported"));
    } catch(...) {
        BOOST_ERROR("Unexpected exception");
    }
    #define __linux__
    #endif

    #ifdef __APPLE__
    #undef __APPLE__
    try {
        boost::compute::context context = boost::compute::opengl_create_shared_context();
    } catch(boost::compute::unsupported_extension_error& error) {
        BOOST_CHECK_EQUAL(std::string(error.what()), std::string("OpenCL extension cl_APPLE_gl_sharing not supported"));
    } catch(...) {
        BOOST_ERROR("Unexpected exception");
    }
    #define __APPLE__
    #endif
}
