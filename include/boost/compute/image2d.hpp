//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_IMAGE2D_HPP
#define BOOST_COMPUTE_IMAGE2D_HPP

#include <vector>

#include <boost/throw_exception.hpp>

#include <boost/compute/config.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/extents.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/image_format.hpp>
#include <boost/compute/memory_object.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

// forward declarations
class command_queue;

class image2d : public memory_object
{
public:
    /// Creates a null image2d object.
    image2d()
        : memory_object()
    {
    }

    /// Creates a new image2d object.
    ///
    /// \see_opencl_ref{clCreateImage}
    image2d(const context &context,
            cl_mem_flags flags,
            const image_format &format,
            size_t image_width,
            size_t image_height,
            size_t image_row_pitch = 0,
            void *host_ptr = 0)
    {
        cl_int error = 0;

        #ifdef CL_VERSION_1_2
        cl_image_desc desc;
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = image_width;
        desc.image_height = image_height;
        desc.image_depth = 1;
        desc.image_array_size = 0;
        desc.image_row_pitch = image_row_pitch;
        desc.image_slice_pitch = 0;
        desc.num_mip_levels = 0;
        desc.num_samples = 0;
        #ifdef CL_VERSION_2_0
        desc.mem_object = 0;
        #else
        desc.buffer = 0;
        #endif

        m_mem = clCreateImage(context,
                              flags,
                              format.get_format_ptr(),
                              &desc,
                              host_ptr,
                              &error);
        #else
        m_mem = clCreateImage2D(context,
                                flags,
                                format.get_format_ptr(),
                                image_width,
                                image_height,
                                image_row_pitch,
                                host_ptr,
                                &error);
        #endif

        if(!m_mem){
            BOOST_THROW_EXCEPTION(opencl_error(error));
        }
    }

    /// Creates a new image2d as a copy of \p other.
    image2d(const image2d &other)
      : memory_object(other)
    {
    }

    /// Copies the image2d from \p other.
    image2d& operator=(const image2d &other)
    {
        memory_object::operator=(other);

        return *this;
    }

    #ifndef BOOST_COMPUTE_NO_RVALUE_REFERENCES
    /// Move-constructs a new image object from \p other.
    image2d(image2d&& other) BOOST_NOEXCEPT
        : memory_object(std::move(other))
    {
    }

    /// Move-assigns the image from \p other to \c *this.
    image2d& operator=(image2d&& other) BOOST_NOEXCEPT
    {
        memory_object::operator=(std::move(other));

        return *this;
    }
    #endif // BOOST_COMPUTE_NO_RVALUE_REFERENCES

    /// Destroys the image2d object.
    ~image2d()
    {
    }

    /// Returns the format for the image.
    image_format get_format() const
    {
        return image_format(get_info<cl_image_format>(CL_IMAGE_FORMAT));
    }

    /// Returns the height of the image.
    size_t height() const
    {
        return get_info<size_t>(CL_IMAGE_HEIGHT);
    }

    /// Returns the width of the image.
    size_t width() const
    {
        return get_info<size_t>(CL_IMAGE_WIDTH);
    }

    /// Returns the size (width, height) of the image.
    extents<2> size() const
    {
        extents<2> size;
        size[0] = get_info<size_t>(CL_IMAGE_WIDTH);
        size[1] = get_info<size_t>(CL_IMAGE_HEIGHT);
        return size;
    }

    /// Returns information about the image.
    ///
    /// \see_opencl_ref{clGetImageInfo}
    template<class T>
    T get_info(cl_image_info info) const
    {
        return detail::get_object_info<T>(clGetImageInfo, m_mem, info);
    }

    size_t get_pixel_count() const
    {
        return height() * width();
    }

    /// Returns the supported image formats for the context.
    ///
    /// \see_opencl_ref{clGetSupportedImageFormats}
    static std::vector<image_format> get_supported_formats(const context &context,
                                                           cl_mem_flags flags)
    {
        cl_uint count = 0;
        clGetSupportedImageFormats(context,
                                   flags,
                                   CL_MEM_OBJECT_IMAGE2D,
                                   0,
                                   0,
                                   &count);

        std::vector<cl_image_format> cl_formats(count);
        clGetSupportedImageFormats(context,
                                   flags,
                                   CL_MEM_OBJECT_IMAGE2D,
                                   count,
                                   &cl_formats[0],
                                   0);

        std::vector<image_format> formats;
        for(size_t i = 0; i < count; i++){
            formats.push_back(image_format(cl_formats[i]));
        }

        return formats;
    }

    /// Creates a new image with a copy of the data in \c *this. Uses \p queue
    /// to perform the copy operation.
    image2d clone(command_queue &queue) const;
};

namespace detail {

// set_kernel_arg specialization for image2d
template<>
struct set_kernel_arg<image2d>
{
    void operator()(kernel &kernel_, size_t index, const image2d &image)
    {
        kernel_.set_arg(index, image.get());
    }
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

BOOST_COMPUTE_TYPE_NAME(boost::compute::image2d, image2d_t)

#endif // BOOST_COMPUTE_IMAGE2D_HPP
