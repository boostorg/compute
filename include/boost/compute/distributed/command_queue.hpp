//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DISTRIBUTED_COMMAND_QUEUE_HPP
#define BOOST_COMPUTE_DISTRIBUTED_COMMAND_QUEUE_HPP

#include <vector>

#include <boost/compute/context.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/distributed/context.hpp>

namespace boost {
namespace compute {
namespace distributed {

class command_queue
{
public:
    enum properties {
        enable_profiling = CL_QUEUE_PROFILING_ENABLE,
        enable_out_of_order_execution = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    };

    enum map_flags {
        map_read = CL_MAP_READ,
        map_write = CL_MAP_WRITE
        #ifdef CL_VERSION_1_2
        ,
        map_write_invalidate_region = CL_MAP_WRITE_INVALIDATE_REGION
        #endif
    };

    /// Creates a null distributed command queue.
    command_queue()
        : m_context(),
          m_queues()
    {

    }

    /// Creates a distributed command queue for all devices in \p context with
    /// \p properties.
    explicit
    command_queue(const ::boost::compute::distributed::context &context,
                  cl_command_queue_properties properties = 0)
        : m_context(context)
    {
        size_t n = m_context.size();
        for(size_t i = 0; i < n; i++) {
            ::boost::compute::context c = m_context.get(i);
            std::vector<device> devices = c.get_devices();
            for(size_t j = 0; j < devices.size(); j++) {
                m_queues.push_back(
                    ::boost::compute::command_queue(c, devices[j], properties)
                );
            }
        }
    }

    /// Creates a distributed command queue containing command queues for each
    /// corresponding device and context from \p devices and \p contexts.
    command_queue(const std::vector< ::boost::compute::context> &contexts,
                  const std::vector< std::vector<device> > &devices,
                  cl_command_queue_properties properties = 0)
    {
        m_context = context(contexts);
        for(size_t i = 0; i < m_context.size(); i++) {
            for(size_t j = 0; j < devices[i].size(); j++) {
                m_queues.push_back(
                    ::boost::compute::command_queue(
                        m_context.get(i), devices[i][j], properties
                    )
                );
            }
        }
    }

    /// Creates a distributed command queue for all devices in \p context.
    command_queue(const ::boost::compute::context &context,
                  cl_command_queue_properties properties = 0)
    {
        m_context = ::boost::compute::distributed::context(context);
        std::vector<device> devices = context.get_devices();
        for(size_t i = 0; i < devices.size(); i++) {
            m_queues.push_back(
                ::boost::compute::command_queue(
                    context, devices[i], properties
                )
            );
        }
    }

    /// Creates a distributed command queue containing \p queues.
    explicit
    command_queue(const std::vector< ::boost::compute::command_queue> queues)
        : m_queues(queues)
    {
        std::vector< ::boost::compute::context> contexts;
        for(size_t i = 0; i < m_queues.size(); i++) {
            contexts.push_back(
                m_queues[i].get_context()
            );
        }
        m_context = context(contexts);
    }

    /// Creates a new command queue object as a copy of \p other.
    command_queue(const command_queue &other)
        : m_context(other.m_context),
          m_queues(other.m_queues)
    {

    }

    /// Copies the command queue object from \p other to \c *this.
    command_queue& operator=(const command_queue &other)
    {
        if(this != &other){
            m_context = other.m_context;
            m_queues = other.m_queues;
        }
        return *this;
    }

    #ifndef BOOST_COMPUTE_NO_RVALUE_REFERENCES
    /// Move-constructs a new command queue object from \p other.
    command_queue(command_queue&& other) BOOST_NOEXCEPT
        : m_context(std::move(other.m_context)),
          m_queues(std::move(other.m_queues))
    {

    }

    /// Move-assigns the command queue from \p other to \c *this.
    command_queue& operator=(command_queue&& other) BOOST_NOEXCEPT
    {
        m_context = std::move(other.m_context);
        m_queues = std::move(other.m_queues);
        return *this;
    }
    #endif // BOOST_COMPUTE_NO_RVALUE_REFERENCES

    /// Returns the number of individual command queues in this
    /// distributed command queue.
    size_t size() const
    {
        return m_queues.size();
    }

    /// Returns the nth command queue.
    ::boost::compute::command_queue& get(size_t n)
    {
        return m_queues[n];
    }

    /// Returns the nth command queue.
    const ::boost::compute::command_queue& get(size_t n) const
    {
        return m_queues[n];
    }

    /// Returns the distributed context used for creating this distributed
    /// command queue.
    const context& get_context() const
    {
        return m_context;
    }

    /// Returns the context of the nth command queue from distributed
    /// command queue.
    ::boost::compute::context get_context(size_t n) const
    {
        return m_queues[n].get_context();
    }

    /// Returns true if all device command queues are in the same OpenCL
    /// context.
    bool one_context() const
    {
        return m_context.one_context();
    }

    /// Returns nth context from command queue's distributed context.
    ::boost::compute::device get_device(size_t n) const
    {
        return m_queues[n].get_device();
    }

    /// Returns \c true if the command queue is the same as \p other.
    bool operator==(const command_queue &other) const
    {
        return (m_context == other.m_context) && (m_queues == other.m_queues);
    }

    /// Returns \c true if the command queue is different from \p other.
    bool operator!=(const command_queue &other) const
    {
        return (m_context != other.m_context) || (m_queues != other.m_queues);
    }

    /// Returns information about nth command queue.
    template<class T>
    T get_info(size_t n, cl_command_queue_info info) const
    {
        return m_queues[n].get_info<T>(info);
    }

    /// Flushes the command queue.
    void flush()
    {
        for(size_t i = 0; i < m_queues.size(); i++)
        {
            m_queues[i].flush();
        }
    }

    /// Blocks until all outstanding commands in the queue have finished.
    void finish()
    {
        for(size_t i = 0; i < m_queues.size(); i++)
        {
            m_queues[i].finish();
        }
    }

private:
    ::boost::compute::distributed::context m_context;
    std::vector< ::boost::compute::command_queue> m_queues;
};


} // end distributed namespace
} // end compute namespace
} // end boost namespace

#endif /* BOOST_COMPUTE_DISTRIBUTED_COMMAND_QUEUE_HPP */
