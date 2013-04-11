//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestImage2D
#include <boost/test/unit_test.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/image2d.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/algorithm/count.hpp>
#include <boost/compute/iterator/detail/pixel_input_iterator.hpp>

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(image2d_get_supported_formats)
{
    bc::context context = bc::system::default_context();

    std::vector<bc::image_format> formats =
        bc::image2d::get_supported_formats(context, bc::image2d::read_only);
    BOOST_CHECK(!formats.empty());
}

BOOST_AUTO_TEST_CASE(get_info)
{
    bc::context context = bc::system::default_context();

    bc::image2d image(
        context,
        bc::image2d::read_only,
        bc::image_format(
            bc::image_format::rgba,
            bc::image_format::unorm_int8
        ),
        48,
        64
    );
    BOOST_CHECK_EQUAL(image.get_info<size_t>(CL_IMAGE_WIDTH), size_t(48));
    BOOST_CHECK_EQUAL(image.get_info<size_t>(CL_IMAGE_HEIGHT), size_t(64));
    BOOST_CHECK_EQUAL(image.get_info<size_t>(CL_IMAGE_DEPTH), size_t(0));
    BOOST_CHECK_EQUAL(image.get_info<size_t>(CL_IMAGE_ROW_PITCH), size_t(48*4));
    BOOST_CHECK_EQUAL(image.get_info<size_t>(CL_IMAGE_SLICE_PITCH), size_t(0));
    BOOST_CHECK_EQUAL(image.get_info<size_t>(CL_IMAGE_ELEMENT_SIZE), size_t(4));

    BOOST_CHECK(bc::image_format(
                    image.get_info<cl_image_format>(CL_IMAGE_FORMAT)) ==
                bc::image_format(
                    bc::image_format::rgba, bc::image_format::unorm_int8));
}

BOOST_AUTO_TEST_CASE(count_with_pixel_iterator)
{
    bc::context context = bc::system::default_context();

    unsigned int data[] = { 0x00000000, 0x000000ff, 0xff0000ff,
                            0xffff00ff, 0x000000ff, 0xff0000ff,
                            0xff0000ff, 0x00ff00ff, 0x0000ffff };

    bc::image2d image(
        context,
        bc::image2d::read_only | bc::image2d::use_host_ptr,
        bc::image_format(
            bc::image_format::rgba,
            bc::image_format::unorm_int8
        ),
        3,
        3,
        3 * 4,
        data
    );

    BOOST_CHECK_EQUAL(
        bc::count(bc::detail::make_pixel_input_iterator<float>(image, 0),
                  bc::detail::make_pixel_input_iterator<float>(image, image.get_pixel_count()),
                  bc::float4_(0, 0, 0, 0)),
        size_t(1));
    BOOST_CHECK_EQUAL(
        bc::count(bc::detail::make_pixel_input_iterator<float>(image, 0),
                  bc::detail::make_pixel_input_iterator<float>(image, image.get_pixel_count()),
                  bc::float4_(1, 0, 0, 0)),
        size_t(2));
    BOOST_CHECK_EQUAL(
        bc::count(bc::detail::make_pixel_input_iterator<float>(image, 0),
                  bc::detail::make_pixel_input_iterator<float>(image, image.get_pixel_count()),
                  bc::float4_(1, 0, 0, 1)),
        size_t(3));
    BOOST_CHECK_EQUAL(
        bc::count(bc::detail::make_pixel_input_iterator<float>(image, 0),
                  bc::detail::make_pixel_input_iterator<float>(image, image.get_pixel_count()),
                  bc::float4_(1, 1, 0, 0)),
        size_t(1));
    BOOST_CHECK_EQUAL(
        bc::count(bc::detail::make_pixel_input_iterator<float>(image, 0),
                  bc::detail::make_pixel_input_iterator<float>(image, image.get_pixel_count()),
                  bc::float4_(1, 0, 1, 1)),
        size_t(1));
}

BOOST_AUTO_TEST_CASE(find_with_pixel_iterator)
{
    bc::context context = bc::system::default_context();

    unsigned int data[] = { 0x00000000, 0x000000ff, 0xff0000ff,
                            0xffff00ff, 0x000000ff, 0xff0000ff,
                            0xff0000ff, 0x00ff00ff, 0x0000ffff };

    bc::image2d image(
        context,
        bc::image2d::read_only | bc::image2d::use_host_ptr,
        bc::image_format(
            bc::image_format::rgba,
            bc::image_format::unorm_int8
        ),
        3,
        3,
        3 * 4,
        data
    );
    BOOST_CHECK_EQUAL(
        std::distance(
            bc::detail::make_pixel_input_iterator<float>(image),
            bc::find(bc::detail::make_pixel_input_iterator<float>(image, 0),
                     bc::detail::make_pixel_input_iterator<float>(image, image.get_pixel_count()),
                     bc::float4_(1, 0, 1, 1))
            ),
        ptrdiff_t(3));
}

// check type_name() for image2d
BOOST_AUTO_TEST_CASE(complex_type_name)
{
    BOOST_CHECK(
        std::strcmp(
            boost::compute::type_name<boost::compute::image2d>(),
            "image2d_t"
        ) == 0
    );
}
