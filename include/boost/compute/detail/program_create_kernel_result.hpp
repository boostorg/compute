//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_PROGRAM_CREATE_KERNEL_RESULT_HPP
#define BOOST_COMPUTE_DETAIL_PROGRAM_CREATE_KERNEL_RESULT_HPP

#include <boost/compute/cl.hpp>

namespace boost {
namespace compute {
namespace detail {

// this is used to break the circular dependency between the kernel
// class and the program::create_kernel() method. kernel has a non-explicit
// constructor taking a program_create_kernel_result which extracts the
// cl_kernel object created by program::create_kernel(). this allows code to
// use program::create_kernel() as if it returned an actual kernel object.
struct program_create_kernel_result
{
    program_create_kernel_result(cl_kernel k) : kernel(k) { }

    cl_kernel kernel;
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_PROGRAM_CREATE_KERNEL_RESULT_HPP
