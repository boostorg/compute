//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestAllocator
#include <boost/test/unit_test.hpp>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/container/allocator.hpp>
#include <boost/compute/container/pinned_allocator.hpp>
#include <boost/compute/container/vector.hpp>

#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(allocate)
{
    boost::compute::allocator<int> allocator(context);

    typedef boost::compute::allocator<int>::pointer pointer;
    pointer x = allocator.allocate(10);
    allocator.deallocate(x, 10);
}

BOOST_AUTO_TEST_CASE(vector_with_pinned_allocator)
{
    boost::compute::vector<int, boost::compute::pinned_allocator<int> > vector;
    vector.push_back(12);
}

BOOST_AUTO_TEST_SUITE_END()
