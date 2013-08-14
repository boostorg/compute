//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestIssue11
#include <boost/test/unit_test.hpp>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <boost/compute/lambda.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types/tuple.hpp>

#include "context_setup.hpp"

namespace compute = boost::compute;

// user-defined data type containing two int's and a float
typedef boost::tuple<int, int, float> UDD;

// function to generate a random UDD on the host
UDD rand_UDD()
{
    int a = rand() % 100;
    int b = rand() % 100;
    float c = (float)(rand() % 100) / 1.3f;

    return boost::make_tuple(a, b, c);
}

// function to compare two UDD's on the host by their first component
bool compare_UDD(const UDD &lhs, const UDD &rhs)
{
    return lhs.get<0>() < rhs.get<0>();
}

BOOST_AUTO_TEST_CASE(issue_11)
{
    using compute::lambda::_1;
    using compute::lambda::_2;
    using compute::lambda::get;

    // create vector of random values on the host
    std::vector<UDD> host_vector(10);
    std::generate(host_vector.begin(), host_vector.end(), rand_UDD);

    // transfer the values to the device
    compute::vector<UDD> device_vector = host_vector;

    // sort values on the device
    compute::sort(
        device_vector.begin(),
        device_vector.end(),
        get<0>(_1) < get<0>(_2),
        queue
    );

    // sort values on the host
    std::sort(
        host_vector.begin(),
        host_vector.end(),
        compare_UDD
    );

    // copy sorted device values back to the host
    std::vector<UDD> tmp(10);
    compute::copy(
        device_vector.begin(),
        device_vector.end(),
        tmp.begin(),
        queue
    );

    // verify sorted values
    for(size_t i = 0; i < host_vector.size(); i++){
        BOOST_CHECK_EQUAL(tmp[i], host_vector[i]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
