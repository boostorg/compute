//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

//[hello_world_example

#include <iostream>
#include <boost/compute.hpp>

int main()
{
    // get the default device
    boost::compute::device device =
        boost::compute::system::default_device();

    // print the GPU's name
    std::cout << "hello from " << device.name() << std::endl;

    return 0;
}
//]
