//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//


#include <iostream>
#include <iterator>

#include <boost/compute.hpp>

// this example shows how to use the max_element() algorithm along with
// a transform_iterator and the length() function to find the longest
// 4-component vector in an array of vectors
int main()
{
    using boost::compute::float4_;

    // vectors data
    float data[] = { 1.0f, 2.0f, 3.0f, 0.0f,
                     4.0f, 5.0f, 6.0f, 0.0f,
                     7.0f, 8.0f, 9.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f };

    // create device vector with the vector data
    boost::compute::vector<float4_> vec(reinterpret_cast<float4_ *>(data),
                                        reinterpret_cast<float4_ *>(data) + 4);

    // find the longest vector
    boost::compute::vector<float4_>::const_iterator iter =
        boost::compute::max_element(
            boost::compute::make_transform_iterator(vec.begin(),
                                                    boost::compute::length<float4_>()),
            boost::compute::make_transform_iterator(vec.end(),
                                                    boost::compute::length<float4_>())
        ).base();

    // print the index of the longest vector
    std::cout << "longest vector index: "
              << std::distance(vec.begin(), iter)
              << std::endl;

    return 0;
}
