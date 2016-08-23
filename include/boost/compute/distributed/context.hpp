//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_CONTEXT_HPP
#define BOOST_COMPUTE_DISTRIBUTED_CONTEXT_HPP

#include <vector>

#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>

namespace boost {
namespace compute {
namespace distributed {

class context
{
public:
    /// Create a null context object.
    context()
        : m_contexts()
    {
    }

    /// Creates a new distributed context for \p devices, containing
    /// boost::compute::context objects, each constructed using corresponding
    /// vector of OpenCL devices and \p properties.
    context(const std::vector< std::vector< ::boost::compute::device> > &devices,
            const std::vector<cl_context_properties*> &properties)
    {
        m_contexts = std::vector< ::boost::compute::context>();
        for(size_t i = 0; i < devices.size(); i++) {
            m_contexts.push_back(
                ::boost::compute::context(devices[i], properties[i])
            );
        }
    }

    /// Creates a new distributed context for \p devices, containing
    /// boost::compute::context objects, each constructed using corresponding
    /// vector of OpenCL devices and default properties.
    explicit context(const std::vector< std::vector< ::boost::compute::device> > &devices)
    {
        m_contexts = std::vector< ::boost::compute::context>();
        for(size_t i = 0; i < devices.size(); i++) {
            m_contexts.push_back(
                ::boost::compute::context(devices[i])
            );
        }
    }

    /// Creates a new distributed context for \p devices with \p properties.
    context(const std::vector< ::boost::compute::device> &devices,
            const std::vector<cl_context_properties*> &properties)
    {
        m_contexts = std::vector< ::boost::compute::context>();
        for(size_t i = 0; i < devices.size(); i++) {
            m_contexts.push_back(
                ::boost::compute::context(devices[i], properties[i])
            );
        }
    }

    /// Creates a new distributed context for \p devices
    explicit context(const std::vector< ::boost::compute::device> &devices)
    {
        m_contexts = std::vector< ::boost::compute::context>();
        for(size_t i = 0; i < devices.size(); i++) {
            m_contexts.push_back(
                ::boost::compute::context(devices[i])
            );
        }
    }

    /// Creates a new distributed context using \p contexts.
    explicit context(const std::vector< ::boost::compute::context>& contexts)
        : m_contexts(contexts)
    {

    }

    /// Creates a new distributed context using contexts from range
    /// [\p first, \p last).
    template <class Iterator>
    explicit context(Iterator first, Iterator last)
        : m_contexts(first, last)
    {

    }

    /// Creates a new distributed context from one \p context.
    explicit context(const ::boost::compute::context& context)
        : m_contexts(1, context)
    {

    }

    /// Creates a new context object as a copy of \p other.
    context(const context &other)
        : m_contexts(other.m_contexts)
    {

    }

    /// Copies the context object from \p other to \c *this.
    context& operator=(const context &other)
    {
        if(this != &other){
            m_contexts =
                std::vector< ::boost::compute::context>(other.m_contexts);
        }
        return *this;
    }

    #ifndef BOOST_COMPUTE_NO_RVALUE_REFERENCES
    /// Move-constructs a new context object from \p other.
    context(context&& other) BOOST_NOEXCEPT
        : m_contexts(std::move(other.m_contexts))
    {

    }

    /// Move-assigns the context from \p other to \c *this.
    context& operator=(context&& other) BOOST_NOEXCEPT
    {
        m_contexts = std::move(other.m_contexts);
        return *this;
    }
    #endif // BOOST_COMPUTE_NO_RVALUE_REFERENCES

    /// Returns number of individual contexts in distributed context.
    size_t size() const
    {
        return m_contexts.size();
    }

    /// Returns nth context.
    const ::boost::compute::context& get(size_t n) const
    {
        return m_contexts[n];
    }

    /// Returns information about nth context.
    template<class T>
    T get_info(size_t n, cl_context_info info) const
    {
        return m_contexts[n].get_info<T>(info);
    }

    /// Returns the device for the nth context in distributed context.
    std::vector<device> get_devices(size_t n) const
    {
        return m_contexts[n].get_info<std::vector<device> >(CL_CONTEXT_DEVICES);
    }

    /// Returns a vector of devices for the distributed context.
    std::vector<std::vector<device> > get_devices() const
    {
        std::vector<std::vector<device> > devices;
        for(size_t i = 0; i < m_contexts.size(); i++) {
            devices.push_back(
                m_contexts[i].get_info<std::vector<device> >(CL_CONTEXT_DEVICES)
            );
        }
        return devices;
    }

    /// Returns \c true if the context is the same as \p other.
    bool operator==(const context &other) const
    {
        return m_contexts == other.m_contexts;
    }

    /// Returns \c true if the context is different from \p other.
    bool operator!=(const context &other) const
    {
        return m_contexts != other.m_contexts;
    }

private:
    std::vector< ::boost::compute::context> m_contexts;
};


inline bool operator==(const ::boost::compute::context &lhs, const context& rhs)
{
    return (rhs.size() == 1) && (rhs.get(0) == lhs);
}

inline bool operator==(const context& lhs, const ::boost::compute::context &rhs)
{
    return (lhs.size() == 1) && (lhs.get(0) == rhs);
}

inline bool operator!=(const ::boost::compute::context &lhs, const context& rhs)
{
    return !(lhs == rhs);
}

inline bool operator!=(const context& lhs, const ::boost::compute::context &rhs)
{
    return !(lhs == rhs);
}

} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_CONTEXT_HPP */
