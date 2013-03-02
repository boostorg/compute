//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestImage3D
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/image3d.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(image3d_get_supported_formats)
{
    bc::context context = bc::system::default_context();

    std::vector<bc::image_format> formats =
        bc::image3d::get_supported_formats(context, bc::image3d::read_only);
}

// check type_name() for image3d
BOOST_AUTO_TEST_CASE(complex_type_name)
{
    BOOST_CHECK(
        std::strcmp(
            boost::compute::type_name<boost::compute::image3d>(),
            "image3d_t"
        ) == 0
    );
}
