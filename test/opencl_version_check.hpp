#ifndef BOOST_COMPUTE_TEST_OPENCL_VERSION_CHECK_HPP
#define BOOST_COMPUTE_TEST_OPENCL_VERSION_CHECK_HPP

#define REQUIRES_OPENCL_VERSION(major, minor) \
    if (!device.check_version(major, minor)) return

#endif
