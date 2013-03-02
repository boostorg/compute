//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_PLATFORM_HPP
#define BOOST_COMPUTE_PLATFORM_HPP

#include <string>
#include <vector>

#include <boost/range/algorithm.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

class platform
{
public:
    explicit platform(cl_platform_id id)
        : m_platform(id)
    {
    }

    platform(const platform &other)
        : m_platform(other.m_platform)
    {
    }

    platform& operator=(const platform &other)
    {
        if(this != &other){
            m_platform = other.m_platform;
        }

        return *this;
    }

    ~platform()
    {
    }

    std::string name() const
    {
        return get_info<std::string>(CL_PLATFORM_NAME);
    }

    std::string vendor() const
    {
        return get_info<std::string>(CL_PLATFORM_VENDOR);
    }

    std::string profile() const
    {
        return get_info<std::string>(CL_PLATFORM_PROFILE);
    }

    std::string version() const
    {
        return get_info<std::string>(CL_PLATFORM_VERSION);
    }

    std::vector<std::string> extensions() const
    {
        std::string extensions_string =
            get_info<std::string>(CL_PLATFORM_EXTENSIONS);
        std::vector<std::string> extensions_vector;
        boost::split(extensions_vector,
                     extensions_string,
                     boost::is_any_of("\t "),
                     boost::token_compress_on);
        return extensions_vector;
    }

    bool supports_extension(const std::string &name) const
    {
        const std::vector<std::string> extensions = this->extensions();

        return boost::find(extensions, name) != extensions.end();
    }

    std::vector<device> devices(cl_device_type type = CL_DEVICE_TYPE_ALL) const
    {
        size_t count = device_count(type);

        std::vector<cl_device_id> device_ids(count);
        cl_int ret = clGetDeviceIDs(m_platform,
                                    type,
                                    static_cast<cl_uint>(count),
                                    &device_ids[0],
                                    0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        std::vector<device> devices;
        for(cl_uint i = 0; i < count; i++){
            devices.push_back(device(device_ids[i]));
        }

        return devices;
    }

    size_t device_count(cl_device_type type = CL_DEVICE_TYPE_ALL) const
    {
        cl_uint count = 0;
        cl_int ret = clGetDeviceIDs(m_platform, type, 0, 0, &count);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return count;
    }

    template<class T>
    T get_info(cl_platform_info info) const
    {
        return detail::get_object_info<T>(clGetPlatformInfo, m_platform, info);
    }

private:
    cl_platform_id m_platform;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_PLATFORM_HPP
