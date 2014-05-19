//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ASYNC_WAIT_HPP
#define BOOST_COMPUTE_ASYNC_WAIT_HPP

#include <boost/compute/config.hpp>
#include <boost/compute/wait_list.hpp>

namespace boost {
namespace compute {

#ifndef BOOST_COMPUTE_DETAIL_NO_VARIADIC_TEMPLATES
/// Blocks until all events have completed. Events can either be event
/// objects or future objects.
template<class... Events>
inline void wait_for_all(Events&&... events)
{
    wait_list l;
    l.insert(std::forward<Events>(events)...);
    l.wait();
}
#endif // BOOST_COMPUTE_DETAIL_NO_VARIADIC_TEMPLATES

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ASYNC_WAIT_HPP
