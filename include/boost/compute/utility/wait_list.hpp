//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_UTILITY_WAIT_LIST_HPP
#define BOOST_COMPUTE_UTILITY_WAIT_LIST_HPP

#include <vector>

#include <boost/compute/event.hpp>

namespace boost {
namespace compute {

template<class T> class future;

/// \class wait_list
/// \brief Stores a list of events.
///
/// The wait_list class stores a set of event objects and can be used to
/// specify dependencies for OpenCL operations or to wait on the host until
/// all of the events have completed.
///
/// This class also provides convenience fnuctions for interacting with
/// OpenCL APIs which typically accept event dependencies as a \c cl_event*
/// pointer and a \c cl_uint size. For example:
/// \code
/// wait_list events = ...;
///
/// clEnqueueNDRangeKernel(..., events.get_event_ptr(), events.size(), ...);
/// \endcode
///
/// \see event, \ref future "future<T>"
class wait_list
{
public:
    /// Creates an empty wait-list.
    wait_list()
    {
    }

    /// Creates a wait-list containing \p event.
    wait_list(const event &event)
    {
        insert(event);
    }

    /// Creates a new wait-list as a copy of \p other.
    wait_list(const wait_list &other)
        : m_events(other.m_events)
    {
    }

    /// Copies the events in the wait-list from \p other.
    wait_list& operator=(const wait_list &other)
    {
        if(this != &other){
            m_events = other.m_events;
        }

        return *this;
    }

    #ifndef BOOST_COMPUTE_NO_RVALUE_REFERENCES
    /// Move-constructs a new wait list object from \p other.
    wait_list(wait_list&& other)
        : m_events(std::move(other.m_events))
    {
    }

    /// Move-assigns the wait list from \p other to \c *this.
    wait_list& operator=(wait_list&& other)
    {
        m_events = std::move(other.m_events);

        return *this;
    }
    #endif // BOOST_COMPUTE_NO_RVALUE_REFERENCES

    /// Destroys the wait-list.
    ~wait_list()
    {
    }

    /// Returns \c true if the wait-list is empty.
    bool empty() const
    {
        return m_events.empty();
    }

    /// Returns the number of events in the wait-list.
    uint_ size() const
    {
        return m_events.size();
    }

    /// Removes all of the events from the wait-list.
    void clear()
    {
        m_events.clear();
    }

    /// Returns a cl_event pointer to the first event in the wait-list.
    /// Returns \c 0 if the wait-list is empty.
    ///
    /// This can be used to pass the wait-list to OpenCL functions which
    /// expect a \c cl_event pointer to refer to a list of events.
    const cl_event* get_event_ptr() const
    {
        if(empty()){
            return 0;
        }

        return reinterpret_cast<const cl_event *>(&m_events[0]);
    }

    /// Inserts \p event into the wait-list.
    void insert(const event &event)
    {
        m_events.push_back(event);
    }

    /// Inserts the event from \p future into the wait-list.
    template<class T>
    void insert(const future<T> &future)
    {
        insert(future.get_event());
    }

    /// Blocks until all of the events in the wait-list have completed.
    ///
    /// Does nothing if the wait-list is empty.
    void wait()
    {
        if(!empty()){
            BOOST_COMPUTE_ASSERT_CL_SUCCESS(
                clWaitForEvents(size(), get_event_ptr())
            );
        }
    }

private:
    std::vector<event> m_events;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_UTILITY_WAIT_LIST_HPP
