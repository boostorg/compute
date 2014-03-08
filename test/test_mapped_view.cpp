//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestMappedView
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/mapped_view.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(fill)
{
    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    for(int i = 0; i < 8; i++){
        BOOST_CHECK_EQUAL(data[i], i+1);
    }

    compute::mapped_view<int> view(data, 8, context);
    compute::fill(view.begin(), view.end(), 4, queue);

    view.map(CL_MAP_READ, queue);
    for(int i = 0; i < 8; i++){
        BOOST_CHECK_EQUAL(data[0], 4);
    }
    view.unmap(queue);
}

BOOST_AUTO_TEST_CASE(sort)
{
    int data[] = { 5, 2, 3, 1, 8, 7, 4, 9 };
    compute::mapped_view<int> view(data, 8, context);

    compute::sort(view.begin(), view.end(), queue);

    view.map(CL_MAP_READ, queue);
    BOOST_CHECK_EQUAL(data[0], 1);
    BOOST_CHECK_EQUAL(data[7], 9);
    view.unmap(queue);
}

BOOST_AUTO_TEST_CASE(reduce)
{
    int data[] = { 5, 2, 3, 1, 8, 7, 4, 9 };
    compute::mapped_view<int> view(data, 8, context);

    int sum = 0;
    compute::reduce(view.begin(), view.end(), &sum, queue);
    BOOST_CHECK_EQUAL(sum, 39);
}

BOOST_AUTO_TEST_SUITE_END()
