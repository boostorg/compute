//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_TIMER_HPP
#define BOOST_COMPUTE_DETAIL_TIMER_HPP

#include <boost/config.hpp>

#ifdef BOOST_COMPUTE_TIMER_USE_STD_CHRONO
#include <chrono>
#else
#include <boost/chrono/chrono.hpp>
#endif

namespace boost {
namespace compute {
namespace detail {

#ifdef BOOST_COMPUTE_TIMER_USE_STD_CHRONO
namespace chrono = std::chrono;
#else
namespace chrono = boost::chrono;
#endif

// a simple timer class based on std::chrono/boost::chrono. the
// elapsed() method returns the time in milliseconds between calls
// of start() and end().
class timer
{
public:
    typedef chrono::high_resolution_clock high_resolution_clock;
    typedef chrono::high_resolution_clock::time_point time_point;

    timer()
    {
        start();
    }

    timer(const timer &other)
        : m_stopped(other.m_stopped),
          m_start_time(other.m_start_time),
          m_stop_time(other.m_stop_time)
    {
    }

    timer& operator=(const timer &other)
    {
        if(this != &other){
            m_stopped = other.m_stopped;
            m_start_time = other.m_start_time;
            m_stop_time = other.m_stop_time;
        }

        return *this;
    }

    ~timer()
    {
    }

    bool is_stopped() const
    {
        return m_stopped;
    }

    size_t elapsed() const
    {
        using chrono::milliseconds;
        using chrono::duration_cast;

        if(m_stopped){
            return duration_cast<milliseconds>(
                m_stop_time - m_start_time
            ).count();
        }
        else {
            time_point now = high_resolution_clock::now();

            return duration_cast<milliseconds>(now - m_start_time).count();
        }
    }

    void start()
    {
        m_start_time = high_resolution_clock::now();
        m_stop_time = time_point();
        m_stopped = false;
    }

    void stop()
    {
        m_stop_time = high_resolution_clock::now();
        m_stopped = true;
    }

private:
    bool m_stopped;
    time_point m_start_time;
    time_point m_stop_time;
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_TIMER_HPP
