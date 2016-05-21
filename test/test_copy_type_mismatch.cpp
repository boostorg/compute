//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

// Undefining BOOST_COMPUTE_USE_OFFLINE_CACHE macro as we want to modify cached
// parameters for copy algorithm without any undesirable consequences (like
// saving modified values of those parameters).
#ifdef BOOST_COMPUTE_USE_OFFLINE_CACHE
    #undef BOOST_COMPUTE_USE_OFFLINE_CACHE
#endif

#define BOOST_TEST_MODULE TestCopyTypeMismatch
#include <boost/test/unit_test.hpp>

#include <list>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <iostream>

#include <boost/compute/svm.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/async/future.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/detail/device_ptr.hpp>
#include <boost/compute/detail/parameter_cache.hpp>

#include "quirks.hpp"
#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;
namespace compute = boost::compute;

BOOST_AUTO_TEST_CASE(is_same_ignore_const)
{
    BOOST_STATIC_ASSERT((
        boost::compute::detail::is_same_value_type<
            std::vector<int>::iterator,
            compute::buffer_iterator<int>
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::compute::detail::is_same_value_type<
            std::vector<int>::const_iterator,
            compute::buffer_iterator<int>
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::compute::detail::is_same_value_type<
            std::vector<int>::iterator,
            compute::buffer_iterator<const int>
        >::value
    ));
    BOOST_STATIC_ASSERT((
        boost::compute::detail::is_same_value_type<
            std::vector<int>::const_iterator,
            compute::buffer_iterator<const int>
        >::value
    ));
}

BOOST_AUTO_TEST_CASE(copy_host_float_to_device_double)
{
    if(!device.supports_extension("cl_khr_fp64")) {
        std::cout << "skipping test: device does not support double" << std::endl;
        return;
    }

    using compute::double_;
    using compute::float_;

    float_ host[] = { 6.1f, 10.2f, 19.3f, 25.4f };
    bc::vector<double_> device_vector(4, context);

    // copy host float data to double device vector
    bc::copy(host, host + 4, device_vector.begin(), queue);
    CHECK_RANGE_EQUAL(double_, 4, device_vector, (6.1f, 10.2f, 19.3f, 25.4f));
}

BOOST_AUTO_TEST_CASE(copy_host_float_to_device_int)
{
    using compute::int_;
    using compute::float_;

    float_ host[] = { 6.1f, -10.2f, 19.3f, 25.4f };
    bc::vector<int_> device_vector(4, context);

    // copy host float data to int device vector
    bc::copy(host, host + 4, device_vector.begin(), queue);
    CHECK_RANGE_EQUAL(
        int_,
        4,
        device_vector,
        (
            static_cast<int_>(6.1f),
            static_cast<int_>(-10.2f),
            static_cast<int_>(19.3f),
            static_cast<int_>(25.4f)
        )
    );
}

BOOST_AUTO_TEST_CASE(copy_host_float_to_device_int_mapping_device_vector)
{
    using compute::int_;
    using compute::uint_;
    using compute::float_;

    float_ host[] = { 6.1f, -10.2f, 19.3f, 25.4f };
    bc::vector<int_> device_vector(4, context);

    std::string cache_key =
        std::string("__boost_compute_copy_to_device_float_int");
    boost::shared_ptr<bc::detail::parameter_cache> parameters =
        bc::detail::parameter_cache::get_global_cache(device);

    // save
    uint_ map_copy_threshold =
        parameters->get(cache_key, "map_copy_threshold", 0);

    // force copy_to_device_map (mapping device vector to the host)
    parameters->set(cache_key, "map_copy_threshold", 1024);

    // copy host float data to int device vector
    bc::copy(host, host + 4, device_vector.begin(), queue);
    CHECK_RANGE_EQUAL(
        int_,
        4,
        device_vector,
        (
            static_cast<int_>(6.1f),
            static_cast<int_>(-10.2f),
            static_cast<int_>(19.3f),
            static_cast<int_>(25.4f)
        )
    );

    // restore
    parameters->set(cache_key, "map_copy_threshold", map_copy_threshold);
}

BOOST_AUTO_TEST_CASE(copy_host_float_to_device_int_convert_on_host)
{
    using compute::int_;
    using compute::uint_;
    using compute::float_;

    std::string cache_key =
        std::string("__boost_compute_copy_to_device_float_int");
    boost::shared_ptr<bc::detail::parameter_cache> parameters =
        bc::detail::parameter_cache::get_global_cache(device);

    // save
    uint_ map_copy_threshold =
        parameters->get(cache_key, "map_copy_threshold", 0);
    uint_ direct_copy_threshold =
        parameters->get(cache_key, "direct_copy_threshold", 0);

    // force copying by casting input data on host and performing
    // normal copy host->device (since types match now)
    parameters->set(cache_key, "map_copy_threshold", 0);
    parameters->set(cache_key, "direct_copy_threshold", 1024);

    float_ host[] = { 6.1f, -10.2f, 19.3f, 25.4f };
    bc::vector<int_> device_vector(4, context);

    // copy host float data to int device vector
    bc::copy(host, host + 4, device_vector.begin(), queue);
    CHECK_RANGE_EQUAL(
        int_,
        4,
        device_vector,
        (
            static_cast<int_>(6.1f),
            static_cast<int_>(-10.2f),
            static_cast<int_>(19.3f),
            static_cast<int_>(25.4f)
        )
    );

    // restore
    parameters->set(cache_key, "map_copy_threshold", map_copy_threshold);
    parameters->set(cache_key, "direct_copy_threshold", direct_copy_threshold);
}

BOOST_AUTO_TEST_CASE(copy_host_float_to_device_int_with_transform)
{
    using compute::int_;
    using compute::uint_;
    using compute::float_;

    std::string cache_key =
        std::string("__boost_compute_copy_to_device_float_int");
    boost::shared_ptr<bc::detail::parameter_cache> parameters =
        bc::detail::parameter_cache::get_global_cache(device);

    // save
    uint_ map_copy_threshold =
        parameters->get(cache_key, "map_copy_threshold", 0);
    uint_ direct_copy_threshold =
        parameters->get(cache_key, "direct_copy_threshold", 0);

    // force copying by mapping input data to the device memory
    // and using transform operation for casting & copying
    parameters->set(cache_key, "map_copy_threshold", 0);
    parameters->set(cache_key, "direct_copy_threshold", 0);

    float_ host[] = { 6.1f, -10.2f, 19.3f, 25.4f };
    bc::vector<int_> device_vector(4, context);

    // copy host float data to int device vector
    bc::copy(host, host + 4, device_vector.begin(), queue);
    CHECK_RANGE_EQUAL(
        int_,
        4,
        device_vector,
        (
            static_cast<int_>(6.1f),
            static_cast<int_>(-10.2f),
            static_cast<int_>(19.3f),
            static_cast<int_>(25.4f)
        )
    );

    // restore
    parameters->set(cache_key, "map_copy_threshold", map_copy_threshold);
    parameters->set(cache_key, "direct_copy_threshold", direct_copy_threshold);
}

BOOST_AUTO_TEST_CASE(copy_async_host_float_to_device_int)
{
    using compute::int_;
    using compute::float_;

    float_ host[] = { 6.1f, -10.2f, 19.3f, 25.4f };
    bc::vector<int_> device_vector(4, context);

    // copy host float data to int device vector
    compute::future<void> future =
        bc::copy_async(host, host + 4, device_vector.begin(), queue);
    future.wait();

    CHECK_RANGE_EQUAL(
        int_,
        4,
        device_vector,
        (
            static_cast<int_>(6.1f),
            static_cast<int_>(-10.2f),
            static_cast<int_>(19.3f),
            static_cast<int_>(25.4f)
        )
    );
}

BOOST_AUTO_TEST_SUITE_END()
