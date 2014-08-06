//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestProgram
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/exception/opencl_error.hpp>
#include <boost/compute/exception/context_error.hpp>
#include <boost/compute/exception/unsupported_extension_error.hpp>

BOOST_AUTO_TEST_CASE(opencl_error_what)
{
    boost::compute::opencl_error error(CL_SUCCESS);
    BOOST_CHECK_EQUAL(std::string(error.what()), std::string("Success"));
}

BOOST_AUTO_TEST_CASE(context_error_what)
{
    boost::compute::context context = boost::compute::system::default_context();
    boost::compute::context_error error(&context, "Test", 0, 0);
    BOOST_CHECK_EQUAL(std::string(error.what()), std::string("Test"));
}

BOOST_AUTO_TEST_CASE(unsupported_extension_error_what)
{
    boost::compute::unsupported_extension_error error("CL_DUMMY_EXTENSION");
    BOOST_CHECK_EQUAL(std::string(error.what()), std::string("OpenCL extension CL_DUMMY_EXTENSION not supported"));
}
