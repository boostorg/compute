//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_IMAGE_SAMPLER_HPP
#define BOOST_COMPUTE_IMAGE_SAMPLER_HPP

#include <boost/throw_exception.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

class image_sampler
{
public:
    enum addressing_mode {
        none = CL_ADDRESS_NONE,
        clamp_to_edge = CL_ADDRESS_CLAMP_TO_EDGE,
        clamp = CL_ADDRESS_CLAMP,
        repeat = CL_ADDRESS_REPEAT
    };

    enum filter_mode {
        nearest = CL_FILTER_NEAREST,
        linear = CL_FILTER_LINEAR
    };

    image_sampler(const context &context,
                  bool normalized_coords,
                  cl_addressing_mode addressing_mode,
                  cl_filter_mode filter_mode)
    {
        cl_int error = 0;
        m_sampler = clCreateSampler(context,
                                    normalized_coords,
                                    addressing_mode,
                                    filter_mode,
                                    &error);
        if(!m_sampler){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }
    }

    explicit image_sampler(cl_sampler sampler, bool retain = true)
        : m_sampler(sampler)
    {
        if(m_sampler && retain){
            clRetainSampler(m_sampler);
        }
    }

    image_sampler(const image_sampler &other)
        : m_sampler(other.m_sampler)
    {
        if(m_sampler){
            clRetainSampler(m_sampler);
        }
    }

    image_sampler& operator=(const image_sampler &other)
    {
        if(this != &other){
            if(m_sampler){
                clReleaseSampler(m_sampler);
            }

            m_sampler = other.m_sampler;

            if(m_sampler){
                clRetainSampler(m_sampler);
            }
        }

        return *this;
    }

    ~image_sampler()
    {
        if(m_sampler){
            clReleaseSampler(m_sampler);
        }
    }

    cl_sampler& get() const
    {
        return const_cast<cl_sampler &>(m_sampler);
    }

    context get_context() const
    {
        return context(get_info<cl_context>(CL_SAMPLER_CONTEXT));
    }

    template<class T>
    T get_info(cl_sampler_info info) const
    {
        return detail::get_object_info<T>(clGetSamplerInfo, m_sampler, info);
    }

    operator cl_sampler() const
    {
        return m_sampler;
    }

private:
    cl_sampler m_sampler;
};

namespace detail {

// type_name() specialization for image_sampler
template<>
struct type_name_trait<image_sampler>
{
    static const char* value()
    {
        return "sampler_t";
    }
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_IMAGE_SAMPLER_HPP
