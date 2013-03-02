//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestKernel
#include <boost/test/unit_test.hpp>

#include <boost/compute/kernel.hpp>
#include <boost/compute/system.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(name)
{
    bc::context context = bc::system::default_context();
    bc::kernel foo = bc::kernel::create_with_source("__kernel void foo(int x) { }",
                                                    "foo",
                                                    context);
    BOOST_CHECK_EQUAL(foo.name(), "foo");

    bc::kernel bar = bc::kernel::create_with_source("__kernel void bar(float x) { }",
                                                    "bar",
                                                    context);
    BOOST_CHECK_EQUAL(bar.name(), "bar");
}

BOOST_AUTO_TEST_CASE(num_args)
{
    bc::context context = bc::system::default_context();
    bc::kernel foo = bc::kernel::create_with_source("__kernel void foo(int x) { }",
                                                    "foo",
                                                    context);
    BOOST_CHECK_EQUAL(foo.num_args(), size_t(1));

    bc::kernel bar = bc::kernel::create_with_source("__kernel void bar(float x, float y) { }",
                                                    "bar",
                                                    context);
    BOOST_CHECK_EQUAL(bar.num_args(), size_t(2));

    bc::kernel baz = bc::kernel::create_with_source("__kernel void baz(char x, char y, char z) { }",
                                                    "baz",
                                                    context);
    BOOST_CHECK_EQUAL(baz.num_args(), size_t(3));
}

BOOST_AUTO_TEST_CASE(get_work_group_info)
{
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);

    const char source[] =
    "__kernel void sum(__global const float *input,\n"
    "                  __global float *output)\n"
    "{\n"
    "    __local float scratch[16];\n"
    "    const uint gid = get_global_id(0);\n"
    "    const uint lid = get_local_id(0);\n"
    "    if(lid < 16)\n"
    "        scratch[lid] = input[gid];\n"
    "}\n";

    boost::compute::program program =
        boost::compute::program::create_with_source(source, context);

    program.build();

    boost::compute::kernel kernel = program.create_kernel("sum");

    using boost::compute::ulong_;

    // check local memory size
    ulong_ local_memory_size =
        kernel.get_work_group_info<ulong_>(device, CL_KERNEL_LOCAL_MEM_SIZE);
    BOOST_CHECK_EQUAL(local_memory_size, ulong_(16 * sizeof(float)));

    // check work group size
    size_t work_group_size =
        kernel.get_work_group_info<size_t>(device, CL_KERNEL_WORK_GROUP_SIZE);
    BOOST_CHECK(work_group_size >= 1);
}

#ifndef BOOST_NO_VARIADIC_TEMPLATES
BOOST_AUTO_TEST_CASE(kernel_set_args)
{
    bc::context context = bc::system::default_context();

    bc::kernel k =
        bc::kernel::create_with_source(
            "__kernel void test(int x, float y, char z) { }",
            "test",
            context
        );

    k.set_args(4, 2.4f, 'a');
}
#endif
