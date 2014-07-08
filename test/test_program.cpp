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
#include <boost/compute/source.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/program.hpp>

#include "context_setup.hpp"

namespace compute = boost::compute;

const char source[] =
    "__kernel void foo(__global float *x, const uint n) { }\n"
    "__kernel void bar(__global int *x, __global int *y) { }\n";

BOOST_AUTO_TEST_CASE(get_program_info)
{
    // create program
    boost::compute::program program =
        boost::compute::program::create_with_source(source, context);

    // build program
    program.build();

    // check program info
#ifndef BOOST_COMPUTE_USE_OFFLINE_CACHE
    BOOST_CHECK(program.source().empty() == false);
#endif
    BOOST_CHECK(program.get_context() == context);
}

BOOST_AUTO_TEST_CASE(create_kernel)
{
    boost::compute::program program =
        boost::compute::program::create_with_source(source, context);
    program.build();

    boost::compute::kernel foo = program.create_kernel("foo");
    boost::compute::kernel bar = program.create_kernel("bar");

    // try to create a kernel that doesn't exist
    BOOST_CHECK_THROW(program.create_kernel("baz"), boost::compute::opencl_error);
}

BOOST_AUTO_TEST_CASE(create_with_binary)
{
    // create program from source
    boost::compute::program source_program =
        boost::compute::program::create_with_source(source, context);
    source_program.build();

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
    binary_program.build();

    // create kernels in binary program
    boost::compute::kernel binary_foo_kernel = binary_program.create_kernel("foo");
    boost::compute::kernel binary_bar_kernel = binary_program.create_kernel("bar");

    // check binary kernels
    BOOST_CHECK_EQUAL(binary_foo_kernel.name(), std::string("foo"));
    BOOST_CHECK_EQUAL(binary_bar_kernel.name(), std::string("bar"));
}

BOOST_AUTO_TEST_CASE(create_with_source_doctest)
{
//! [create_with_source]
std::string source = "__kernel void foo(__global int *data) { }";

boost::compute::program foo_program =
    boost::compute::program::create_with_source(source, context);
//! [create_with_source]

    foo_program.build();
}

#ifdef CL_VERSION_1_2
BOOST_AUTO_TEST_CASE(compile_and_link)
{
    // create the library program
    const char library_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        // generic square function definition
        T square(T x) { return x * x; }
    );

    compute::program library_program =
        compute::program::create_with_source(library_source, context);

    library_program.compile("-DT=int");

    // create the kernel program
    const char kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        // forward declare square function
        extern int square(int);

        // square kernel definition
        __kernel void square_kernel(__global int *x)
        {
            x[0] = square(x[0]);
        }
    );

    compute::program square_program =
        compute::program::create_with_source(kernel_source, context);

    square_program.compile();

    // link the programs
    std::vector<compute::program> programs;
    programs.push_back(library_program);
    programs.push_back(square_program);

    compute::program linked_program =
        compute::program::link(programs, context);

    // create the square kernel
    compute::kernel square_kernel =
        linked_program.create_kernel("square_kernel");
    BOOST_CHECK_EQUAL(square_kernel.name(), "square_kernel");
}
#endif // CL_VERSION_1_2

BOOST_AUTO_TEST_SUITE_END()
