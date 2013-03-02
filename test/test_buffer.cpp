//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestBuffer
#include <boost/test/unit_test.hpp>

#include <boost/compute/buffer.hpp>
#include <boost/compute/system.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(size)
{
    bc::context context = bc::system::default_context();
    bc::buffer buffer(context, 100);
    BOOST_CHECK_EQUAL(buffer.size(), size_t(100));
    BOOST_VERIFY(buffer.max_size() > buffer.size());
}

BOOST_AUTO_TEST_CASE(context)
{
    bc::context context = bc::system::default_context();
    bc::buffer buffer(context, 100);
    BOOST_VERIFY(buffer.get_context() == context);
}

BOOST_AUTO_TEST_CASE(equality_operator)
{
    bc::context context = bc::system::default_context();
    bc::buffer a(context, 10);
    bc::buffer b(context, 10);
    BOOST_VERIFY(a == a);
    BOOST_VERIFY(b == b);
    BOOST_VERIFY(!(a == b));
    BOOST_VERIFY(a != b);

    a = b;
    BOOST_VERIFY(a == b);
    BOOST_VERIFY(!(a != b));
}
