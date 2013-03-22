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

#include <boost/compute/kernel.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/program.hpp>

const char source[] =
    "__kernel void foo(__global float *x, const uint n) { }\n"
    "__kernel void bar(__global int *x, __global int *y) { }\n";

BOOST_AUTO_TEST_CASE(get_program_info)
{
    boost::compute::context context =
        boost::compute::system::default_context();

    // create program
    boost::compute::program program =
        boost::compute::program::create_with_source(source, context);

    // check program info
    BOOST_CHECK(program.source().empty() == false);
    BOOST_CHECK(program.get_context() == context);
}

BOOST_AUTO_TEST_CASE(create_kernel)
{
    boost::compute::context context =
        boost::compute::system::default_context();

    boost::compute::program program =
        boost::compute::program::create_with_source(source, context);

    boost::compute::kernel foo = program.create_kernel("foo");
    boost::compute::kernel bar = program.create_kernel("bar");
}

BOOST_AUTO_TEST_CASE(create_with_binary)
{
    boost::compute::context context =
        boost::compute::system::default_context();

    // create program from source
    boost::compute::program source_program =
        boost::compute::program::create_with_source(source, context);

    // create kernels in source program
    boost::compute::kernel source_foo_kernel = source_program.create_kernel("foo");
    boost::compute::kernel source_bar_kernel = source_program.create_kernel("bar");

    // check source kernels
    BOOST_CHECK_EQUAL(source_foo_kernel.name(), std::string("foo"));
    BOOST_CHECK_EQUAL(source_bar_kernel.name(), std::string("bar"));

    // get binary
    std::vector<unsigned char> binary = source_program.binary();

    // create program from binary
    boost::compute::program binary_program =
        boost::compute::program::create_with_binary(binary, context);

    // create kernels in binary program
    boost::compute::kernel binary_foo_kernel = binary_program.create_kernel("foo");
    boost::compute::kernel binary_bar_kernel = binary_program.create_kernel("bar");

    // check binary kernels
    BOOST_CHECK_EQUAL(binary_foo_kernel.name(), std::string("foo"));
    BOOST_CHECK_EQUAL(binary_bar_kernel.name(), std::string("bar"));
}
