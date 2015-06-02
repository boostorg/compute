//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestSvmPtr
#include <boost/test/unit_test.hpp>

#include <boost/compute/core.hpp>
#include <boost/compute/svm.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(empty)
{
}

#ifdef CL_VERSION_2_0
BOOST_AUTO_TEST_CASE(alloc)
{
    compute::svm_ptr<int> ptr = compute::svm_alloc<int>(context, 8);
    compute::svm_free(context, ptr);
}

BOOST_AUTO_TEST_CASE(sum_svm_kernel)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void sum_svm_mem(__global const int *ptr, __global int *result)
        {
            int sum = 0;
            for(uint i = 0; i < 8; i++){
                sum += ptr[i];
            }
            *result = sum;
        }
    );

    compute::program program =
        compute::program::build_with_source(source, context);

    compute::kernel sum_svm_mem_kernel = program.create_kernel("sum_svm_mem");

    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    compute::svm_ptr<int> ptr = compute::svm_alloc<int>(context, 8);
    queue.enqueue_svm_memcpy(ptr.get(), data, 8 * sizeof(int));

    compute::vector<int> result(1, context);

    sum_svm_mem_kernel.set_arg(0, ptr);
    sum_svm_mem_kernel.set_arg(1, result);
    queue.enqueue_task(sum_svm_mem_kernel);

    CHECK_RANGE_EQUAL(int, 1, result, (36));

    compute::svm_free(context, ptr);
}
#endif // CL_VERSION_2_0

BOOST_AUTO_TEST_SUITE_END()
