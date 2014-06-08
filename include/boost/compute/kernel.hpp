//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_KERNEL_HPP
#define BOOST_COMPUTE_KERNEL_HPP

#include <string>

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/move/move.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/type_traits/is_fundamental.hpp>
#include <boost/compute/detail/get_object_info.hpp>
#include <boost/compute/detail/assert_cl_success.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class T> struct set_kernel_arg;

} // end detail namespace

/// \class kernel
/// \brief A compute kernel.
///
/// \see command_queue, program
class kernel
{
public:
    /// Creates a null kernel object.
    kernel()
        : m_kernel(0)
    {
    }

    /// Creates a new kernel object for \p kernel. If \p retain is
    /// \c true, the reference count for \p kernel will be incremented.
    explicit kernel(cl_kernel kernel, bool retain = true)
        : m_kernel(kernel)
    {
        if(m_kernel && retain){
            clRetainKernel(m_kernel);
        }
    }

    /// Creates a new kernel object with \p name from \p program.
    kernel(const program &program, const std::string &name)
    {
        cl_int error = 0;
        m_kernel = clCreateKernel(program.get(), name.c_str(), &error);

        if(!m_kernel){
            BOOST_THROW_EXCEPTION(opencl_error(error));
        }
    }

    /// Creates a new kernel object as a copy of \p other.
    kernel(const kernel &other)
        : m_kernel(other.m_kernel)
    {
        if(m_kernel){
            clRetainKernel(m_kernel);
        }
    }

    /// \internal_
    kernel(BOOST_RV_REF(kernel) other)
        : m_kernel(other.m_kernel)
    {
        other.m_kernel = 0;
    }

    /// Copies the kernel object from \p other to \c *this.
    kernel& operator=(const kernel &other)
    {
        if(this != &other){
            if(m_kernel){
                clReleaseKernel(m_kernel);
            }

            m_kernel = other.m_kernel;

            if(m_kernel){
                clRetainKernel(m_kernel);
            }
        }

        return *this;
    }

    /// \internal_
    kernel& operator=(BOOST_RV_REF(kernel) other)
    {
        if(this != &other){
            if(m_kernel){
                clReleaseKernel(m_kernel);
            }

            m_kernel = other.m_kernel;
            other.m_kernel = 0;
        }

        return *this;
    }

    /// Destroys the kernel object.
    ~kernel()
    {
        if(m_kernel){
            BOOST_COMPUTE_ASSERT_CL_SUCCESS(
                clReleaseKernel(m_kernel)
            );
        }
    }

    /// Returns a reference to the underlying OpenCL kernel object.
    cl_kernel& get() const
    {
        return const_cast<cl_kernel &>(m_kernel);
    }

    /// Returns the function name for the kernel.
    std::string name() const
    {
        return get_info<std::string>(CL_KERNEL_FUNCTION_NAME);
    }

    /// Returns the number of arguments for the kernel.
    size_t arity() const
    {
        return get_info<cl_uint>(CL_KERNEL_NUM_ARGS);
    }

    /// Returns the program for the kernel.
    program get_program() const
    {
        return program(get_info<cl_program>(CL_KERNEL_PROGRAM));
    }

    /// Returns the context for the kernel.
    context get_context() const
    {
        return context(get_info<cl_context>(CL_KERNEL_CONTEXT));
    }

    /// Returns information about the kernel.
    ///
    /// \see_opencl_ref{clGetKernelInfo}
    template<class T>
    T get_info(cl_kernel_info info) const
    {
        return detail::get_object_info<T>(clGetKernelInfo, m_kernel, info);
    }

    #if defined(CL_VERSION_1_2) || defined(BOOST_COMPUTE_DOXYGEN_INVOKED)
    /// Returns information about the argument at \p index.
    ///
    /// \opencl_version_warning{1,2}
    ///
    /// \see_opencl_ref{clGetKernelArgInfo}
    template<class T>
    T get_arg_info(size_t index, cl_kernel_arg_info info)
    {
        T value;
        cl_int ret = clGetKernelArgInfo(m_kernel,
                                        index,
                                        info,
                                        sizeof(T),
                                        &value,
                                        0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(opencl_error(ret));
        }

        return value;
    }
    #endif // CL_VERSION_1_2

    /// Returns work-group information for the kernel with \p device.
    ///
    /// \see_opencl_ref{clGetKernelWorkGroupInfo}
    template<class T>
    T get_work_group_info(const device &device, cl_kernel_work_group_info info)
    {
        T value;
        cl_int ret = clGetKernelWorkGroupInfo(m_kernel,
                                              device.id(),
                                              info,
                                              sizeof(T),
                                              &value,
                                              0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(opencl_error(ret));
        }

        return value;
    }

    /// Sets the argument at \p index to \p value with \p size.
    ///
    /// \see_opencl_ref{clSetKernelArg}
    void set_arg(size_t index, size_t size, const void *value)
    {
        BOOST_ASSERT(index < arity());

        cl_int ret = clSetKernelArg(m_kernel,
                                    static_cast<cl_uint>(index),
                                    size,
                                    value);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(opencl_error(ret));
        }
    }

    /// Sets the argument at \p index to \p value.
    ///
    /// For built-in types (e.g. \c float, \c int4_), this is equivalent to
    /// calling set_arg(index, sizeof(type), &value).
    ///
    /// Additionally, this method is specialized for device memory objects
    /// such as buffer and image2d. This allows for them to be passed directly
    /// without having to extract their underlying cl_mem object.
    ///
    /// This method is also specialized for device container types such as
    /// vector<T> and array<T, N>. This allows for them to be passed directly
    /// as kernel arguments without having to extract their underlying buffer.
    template<class T>
    void set_arg(size_t index, const T &value)
    {
        // if you get a compilation error pointing here it means you
        // attempted to set a kernel argument from an invalid type.
        detail::set_kernel_arg<T>()(*this, index, value);
    }

    /// \internal_
    void set_arg(size_t index, const cl_mem mem)
    {
        set_arg(index, sizeof(cl_mem), static_cast<const void *>(&mem));
    }

    /// \internal_
    void set_arg(size_t index, const cl_sampler sampler)
    {
        set_arg(index, sizeof(cl_sampler), static_cast<const void *>(&sampler));
    }

    #ifndef BOOST_NO_VARIADIC_TEMPLATES
    /// Sets the arguments for the kernel to \p args.
    template<class... T>
    void set_args(T&&... args)
    {
        BOOST_ASSERT(sizeof...(T) <= arity());

        _set_args<0>(args...);
    }
    #endif // BOOST_NO_VARIADIC_TEMPLATES

    /// \internal_
    operator cl_kernel() const
    {
        return m_kernel;
    }

    /// \internal_
    static kernel create_with_source(const std::string &source,
                                     const std::string &name,
                                     const context &context)
    {
        return program::build_with_source(source, context).create_kernel(name);
    }

private:
    #ifndef BOOST_NO_VARIADIC_TEMPLATES
    /// \internal_
    template<size_t N>
    void _set_args()
    {
    }

    /// \internal_
    template<size_t N, class T, class... Args>
    void _set_args(T&& arg, Args&&... rest)
    {
        set_arg(N, arg);
        _set_args<N+1>(rest...);
    }
    #endif // BOOST_NO_VARIADIC_TEMPLATES

private:
    BOOST_COPYABLE_AND_MOVABLE(kernel)

    cl_kernel m_kernel;
};

inline kernel program::create_kernel(const std::string &name) const
{
    return kernel(*this, name);
}

namespace detail {

// set_kernel_arg implementation for built-in types
template<class T>
struct set_kernel_arg
{
    typename boost::enable_if<is_fundamental<T> >::type
    operator()(kernel &kernel_, size_t index, const T &value)
    {
        kernel_.set_arg(index, sizeof(T), &value);
    }
};

// set_kernel_arg specialization for char (different from built-in cl_char)
template<>
struct set_kernel_arg<char>
{
    void operator()(kernel &kernel_, size_t index, const char c)
    {
        kernel_.set_arg(index, sizeof(char), &c);
    }
};

} // end detail namespace
} // end namespace compute
} // end namespace boost

#endif // BOOST_COMPUTE_KERNEL_HPP
