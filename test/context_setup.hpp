#ifndef BOOST_COMPUTE_TEST_CONTEXT_SETUP_HPP
#define BOOST_COMPUTE_TEST_CONTEXT_SETUP_HPP

#include <boost/compute/command_queue.hpp>

struct Context {
    static boost::compute::device        device;
    static boost::compute::context       context;
    static boost::compute::command_queue queue;

    Context() {
        device  = boost::compute::system::default_device();
        context = boost::compute::system::default_context();
        queue   = boost::compute::command_queue(context, device);

        std::cout << device.name() << std::endl;
    }
};

boost::compute::device        Context::device;
boost::compute::context       Context::context;
boost::compute::command_queue Context::queue;

struct ContextRef {
    boost::compute::device        &device;
    boost::compute::context       &context;
    boost::compute::command_queue &queue;

    ContextRef() :
        device ( Context::device  ),
        context( Context::context ),
        queue  ( Context::queue   )
    {}
};

BOOST_GLOBAL_FIXTURE( Context )
BOOST_FIXTURE_TEST_SUITE(compute_test, ContextRef)

#endif
