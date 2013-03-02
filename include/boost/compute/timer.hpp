//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_TIMER_HPP
#define BOOST_COMPUTE_TIMER_HPP

#include <boost/compute/event.hpp>
#include <boost/compute/command_queue.hpp>

namespace boost {
namespace compute {

class timer
{
public:
    explicit timer(const command_queue &queue)
        : m_queue(queue)
    {
        start();
    }

    timer(const timer &other)
        : m_queue(other.m_queue),
          m_start_event(other.m_start_event),
          m_stop_event(other.m_stop_event)
    {
    }

    timer& operator=(const timer &other)
    {
        if(this != &other){
            m_queue = other.m_queue;
            m_start_event = other.m_start_event;
            m_stop_event = other.m_stop_event;
        }

        return *this;
    }

    ~timer()
    {
    }

    bool is_supported() const
    {
        return m_queue.get_properties() & command_queue::enable_profiling;
    }

    bool is_stopped() const
    {
        return static_cast<bool>(m_stop_event);
    }

    ulong_ elapsed() const
    {
        if(!is_supported() || !m_start_event){
            return 0;
        }

        if(m_stop_event){
            const_cast<event &>(m_stop_event).wait();

            return get_start_time(m_stop_event) - get_end_time(m_start_event);
        }
        else {
            event now = const_cast<command_queue &>(m_queue).enqueue_marker();
            now.wait();

            return get_start_time(now) - get_end_time(m_start_event);
        }
    }

    void start()
    {
        m_start_event = m_queue.enqueue_marker();

        m_stop_event = event();
    }

    void stop()
    {
        m_stop_event = m_queue.enqueue_marker();
    }

private:
    ulong_ get_start_time(const event &event) const
    {
        return event.get_profiling_info<ulong_>(event::profiling_command_start);
    }

    ulong_ get_end_time(const event &event) const
    {
        return event.get_profiling_info<ulong_>(event::profiling_command_end);
    }

private:
    command_queue m_queue;
    event m_start_event;
    event m_stop_event;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_TIMER_HPP
