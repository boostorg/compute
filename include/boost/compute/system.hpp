//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_SYSTEM_HPP
#define BOOST_COMPUTE_SYSTEM_HPP

#include <string>
#include <vector>
#include <cstdlib>

#include <boost/foreach.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/platform.hpp>

namespace boost {
namespace compute {

class system
{
public:
    static device default_device()
    {
        // check for device from environment variable
        const char *name = std::getenv("BOOST_COMPUTE_DEFAULT_DEVICE");
        if(name){
            device device = find_device(name);
            if(device.id()){
                return device;
            }
        }

        // check for a gpu device
        device gpu = default_gpu_device();
        if(gpu.id() != 0){
            return gpu;
        }

        // check for a cpu device
        device cpu = default_cpu_device();
        if(cpu.id() != 0){
            return cpu;
        }

        // return a null device
        return device();
    }

    static device default_cpu_device()
    {
        BOOST_FOREACH(const device &device, devices()){
            if(device.type() == device::cpu){
                return device;
            }
        }

        return device();
    }

    static device default_gpu_device()
    {
        BOOST_FOREACH(const device &device, devices()){
            if(device.type() == device::gpu){
                return device;
            }
        }

        return device();
    }

    static device find_device(const std::string &name)
    {
        BOOST_FOREACH(const device &device, devices()){
            if(device.name().find(name) != std::string::npos){
                return device;
            }
        }

        return device();
    }

    static std::vector<device> devices()
    {
        std::vector<device> devices;

        BOOST_FOREACH(const platform &platform, platforms()){
            BOOST_FOREACH(const device &device, platform.devices()){
                devices.push_back(device);
            }
        }

        return devices;
    }

    static size_t device_count()
    {
        size_t count = 0;

        BOOST_FOREACH(const platform &platform, platforms()){
            count += platform.device_count();
        }

        return count;
    }

    static context default_context()
    {
        static context default_context(default_device());

        return default_context;
    }

    static std::vector<platform> platforms()
    {
        cl_uint count = 0;
        clGetPlatformIDs(0, 0, &count);

        std::vector<cl_platform_id> platform_ids(count);
        clGetPlatformIDs(count, &platform_ids[0], 0);

        std::vector<platform> platforms;
        for(size_t i = 0; i < platform_ids.size(); i++){
            platforms.push_back(platform(platform_ids[i]));
        }

        return platforms;
    }

    static size_t platform_count()
    {
        cl_uint count = 0;
        clGetPlatformIDs(0, 0, &count);
        return static_cast<size_t>(count);
    }
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_SYSTEM_HPP
