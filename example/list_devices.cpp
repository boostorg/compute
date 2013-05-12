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
#include <boost/compute.hpp>

int main()
{
    std::vector<boost::compute::platform>
        platforms = boost::compute::system::platforms();

    for(size_t i = 0; i < platforms.size(); i++){
        const boost::compute::platform &platform = platforms[i];

        std::cout << "Platform '" << platform.name() << "'" << std::endl;

        std::vector<boost::compute::device> devices = platform.devices();
        for(size_t j = 0; j < devices.size(); j++){
            const boost::compute::device &device = devices[j];

            std::string type;
            if(device.type() == boost::compute::device::gpu)
                type = "GPU Device";
            else if(device.type() == boost::compute::device::cpu)
                type = "CPU Device";
            else if(device.type() == boost::compute::device::accelerator)
                type = "Accelerator Device";
            else
                type = "Unknown Device";

            std::cout << "  " << type << ": " << device.name() << std::endl;
        }
    }

    return 0;
}
