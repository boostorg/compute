//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_IMAGE_FORMAT_HPP
#define BOOST_COMPUTE_IMAGE_FORMAT_HPP

#include <boost/compute/cl.hpp>

namespace boost {
namespace compute {

class image_format
{
public:
    enum channel_order {
        r = CL_R,
        a = CL_A,
        intensity = CL_INTENSITY,
        luminance = CL_LUMINANCE,
        rg = CL_RG,
        ra = CL_RA,
        rgb = CL_RGB,
        rgba = CL_RGBA,
        argb = CL_ARGB,
        bgra = CL_BGRA
    };

    enum channel_data_type {
        snorm_int8 = CL_SNORM_INT8,
        snorm_int16 = CL_SNORM_INT16,
        unorm_int8 = CL_UNORM_INT8,
        unorm_int16 = CL_UNORM_INT16,
        unorm_short_565 = CL_UNORM_SHORT_565,
        unorm_short_555 = CL_UNORM_SHORT_555,
        unorm_int_101010 = CL_UNORM_INT_101010,
        signed_int8 = CL_SIGNED_INT8,
        signed_int16 = CL_SIGNED_INT16,
        signed_int32 = CL_SIGNED_INT32,
        unsigned_int8 = CL_UNSIGNED_INT8,
        unsigned_int16 = CL_UNSIGNED_INT16,
        unsigned_int32 = CL_UNSIGNED_INT32,
        float16 = CL_HALF_FLOAT,
        float32 = CL_FLOAT
    };

    image_format()
    {
    }

    image_format(cl_channel_order order, cl_channel_type type)
    {
        m_format.image_channel_order = order;
        m_format.image_channel_data_type = type;
    }

    explicit image_format(const cl_image_format &format)
    {
        m_format.image_channel_order = format.image_channel_order;
        m_format.image_channel_data_type = format.image_channel_data_type;
    }

    image_format(const image_format &other)
        : m_format(other.m_format)
    {
    }

    image_format& operator=(const image_format &other)
    {
        if(this != &other){
            m_format = other.m_format;
        }

        return *this;
    }

    ~image_format()
    {
    }

    const cl_image_format* get_format_ptr() const
    {
        return &m_format;
    }

    bool operator==(const image_format &other) const
    {
        return m_format.image_channel_order ==
                   other.m_format.image_channel_order &&
               m_format.image_channel_data_type ==
                   other.m_format.image_channel_data_type;
    }

    bool operator!=(const image_format &other) const
    {
        return !(*this == other);
    }

private:
    cl_image_format m_format;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_IMAGE_FORMAT_HPP
