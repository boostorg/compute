//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DEVICE_HPP
#define BOOST_COMPUTE_DEVICE_HPP

#include <string>
#include <vector>

#include <boost/move/move.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

class device
{
public:
    enum type {
        cpu = CL_DEVICE_TYPE_CPU,
        gpu = CL_DEVICE_TYPE_GPU,
        accelerator = CL_DEVICE_TYPE_ACCELERATOR
    };

    device()
        : m_id(0)
    {
    }

    explicit device(cl_device_id id, bool retain = true)
        : m_id(id)
    {
        #ifdef CL_VERSION_1_2
        if(m_id && retain){
            clRetainDevice(m_id);
        }
        #else
        (void) retain;
        #endif
    }

    device(const device &other)
        : m_id(other.m_id)
    {
        #ifdef CL_VERSION_1_2
        if(m_id){
            clRetainDevice(m_id);
        }
        #endif
    }

    device(BOOST_RV_REF(device) other)
        : m_id(other.m_id)
    {
        other.m_id = 0;
    }

    device& operator=(const device &other)
    {
        if(this != &other){
            #ifdef CL_VERSION_1_2
            if(m_id){
                clReleaseDevice(m_id);
            }
            #endif

            m_id = other.m_id;

            #ifdef CL_VERSION_1_2
            if(m_id){
                clRetainDevice(m_id);
            }
            #endif
        }

        return *this;
    }

    device& operator=(BOOST_RV_REF(device) other)
    {
        if(this != &other){
            #ifdef CL_VERSION_1_2
            if(m_id){
                clReleaseDevice(m_id);
            }
            #endif

            m_id = other.m_id;
            other.m_id = 0;
        }

        return *this;
    }

    ~device()
    {
        #ifdef CL_VERSION_1_2
        if(m_id){
            clReleaseDevice(m_id);
        }
        #endif
    }

    cl_device_id id() const
    {
        return m_id;
    }

    cl_device_type type() const
    {
        return get_info<cl_device_type>(CL_DEVICE_TYPE);
    }

    std::string name() const
    {
        return get_info<std::string>(CL_DEVICE_NAME);
    }

    std::string vendor() const
    {
        return get_info<std::string>(CL_DEVICE_VENDOR);
    }

    std::string profile() const
    {
        return get_info<std::string>(CL_DEVICE_PROFILE);
    }

    std::string version() const
    {
        return get_info<std::string>(CL_DEVICE_VERSION);
    }

    std::string driver_version() const
    {
        return get_info<std::string>(CL_DRIVER_VERSION);
    }

    std::vector<std::string> extensions() const
    {
        std::string extensions_string =
            get_info<std::string>(CL_DEVICE_EXTENSIONS);
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

    uint_ address_bits() const
    {
        return get_info<uint_>(CL_DEVICE_ADDRESS_BITS);
    }

    ulong_ global_memory_size() const
    {
        return get_info<ulong_>(CL_DEVICE_GLOBAL_MEM_SIZE);
    }

    ulong_ local_memory_size() const
    {
        return get_info<ulong_>(CL_DEVICE_LOCAL_MEM_SIZE);
    }

    uint_ clock_frequency() const
    {
        return get_info<uint_>(CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    uint_ compute_units() const
    {
        return get_info<uint_>(CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    ulong_ max_memory_alloc_size() const
    {
        return get_info<ulong_>(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    }

    size_t max_work_group_size() const
    {
        return get_info<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    uint_ max_work_item_dimensions() const
    {
        return get_info<uint_>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    }

    template<class T>
    uint_ preferred_vector_width() const
    {
        return 0;
    }

    size_t profiling_timer_resolution() const
    {
        return get_info<size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION);
    }

    template<class T>
    T get_info(cl_device_info info) const
    {
        return detail::get_object_info<T>(clGetDeviceInfo, m_id, info);
    }

    #ifdef CL_VERSION_1_2
    std::vector<device>
    partition(const cl_device_partition_property *properties) const
    {
        // get sub-device count
        uint_ count = 0;
        int_ ret = clCreateSubDevices(m_id, properties, 0, 0, &count);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        // get sub-device ids
        std::vector<cl_device_id> ids(count);
        ret = clCreateSubDevices(m_id, properties, count, &ids[0], 0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        // convert ids to device objects
        std::vector<device> devices(count);
        for(size_t i = 0; i < count; i++){
            devices[i] = device(ids[i], false);
        }

        return devices;
    }

    std::vector<device> partition_equally(size_t count) const
    {
        cl_device_partition_property properties[] =
            { CL_DEVICE_PARTITION_EQUALLY, count, 0 };

        return partition(properties);
    }

    std::vector<device>
    partition_by_counts(const std::vector<size_t> &counts) const
    {
        std::vector<cl_device_partition_property> properties;

        properties.push_back(CL_DEVICE_PARTITION_BY_COUNTS);
        for(size_t i = 0; i < counts.size(); i++){
            properties.push_back(
                static_cast<cl_device_partition_property>(counts[i]));
        }
        properties.push_back(CL_DEVICE_PARTITION_BY_COUNTS_LIST_END);
        properties.push_back(0);

        return partition(&properties[0]);
    }

    std::vector<device>
    partition_by_affinity_domain(cl_device_affinity_domain domain) const
    {
        cl_device_partition_property properties[] =
            { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, domain, 0 };

        return partition(properties);
    }
    #endif // CL_VERSION_1_2

private:
    BOOST_COPYABLE_AND_MOVABLE(device)

    cl_device_id m_id;
};

template<>
inline uint_ device::preferred_vector_width<short_>() const
{
    return get_info<uint_>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
}

template<>
inline uint_ device::preferred_vector_width<int_>() const
{
    return get_info<uint_>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
}

template<>
inline uint_ device::preferred_vector_width<long_>() const
{
    return get_info<uint_>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
}

template<>
inline uint_ device::preferred_vector_width<float_>() const
{
    return get_info<uint_>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
}

template<>
inline uint_ device::preferred_vector_width<double_>() const
{
    return get_info<uint_>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DEVICE_HPP
