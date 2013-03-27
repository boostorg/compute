//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestDevice
#include <boost/test/unit_test.hpp>

#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>

BOOST_AUTO_TEST_CASE(null_device)
{
    boost::compute::device null;
    BOOST_CHECK(null.id() == cl_device_id());
}

BOOST_AUTO_TEST_CASE(get_gpu_type)
{
    boost::compute::device gpu = boost::compute::system::default_gpu_device();
    if(gpu.id()){
        BOOST_CHECK(gpu.type() == boost::compute::device::gpu);
    }
}

BOOST_AUTO_TEST_CASE(get_cpu_type)
{
    boost::compute::device cpu = boost::compute::system::default_cpu_device();
    if(cpu.id()){
        BOOST_CHECK(cpu.type() == boost::compute::device::cpu);
    }
}

BOOST_AUTO_TEST_CASE(get_gpu_name)
{
    boost::compute::device gpu = boost::compute::system::default_gpu_device();
    if(gpu.id()){
        BOOST_CHECK(!gpu.name().empty());
    }
}

BOOST_AUTO_TEST_CASE(get_max_work_item_sizes)
{
    boost::compute::device device = boost::compute::system::default_device();

    std::vector<size_t> max_work_item_sizes =
        device.get_info<std::vector<size_t> >(CL_DEVICE_MAX_WORK_ITEM_SIZES);
    BOOST_CHECK_GE(max_work_item_sizes.size(), size_t(3));
    BOOST_CHECK_GE(max_work_item_sizes[0], size_t(1));
    BOOST_CHECK_GE(max_work_item_sizes[1], size_t(1));
    BOOST_CHECK_GE(max_work_item_sizes[2], size_t(1));
}
