//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_EXTREMA_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_EXTREMA_HPP

#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/algorithm/detail/find_extrema_with_atomics.hpp>
#include <boost/compute/algorithm/detail/serial_find_extrema.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator>
inline InputIterator find_extrema(InputIterator first,
                                  InputIterator last,
                                  char sign,
                                  command_queue &queue)
{
    size_t count = iterator_range_size(first, last);

    // handle trivial cases
    if(count == 0 || count == 1){
        return first;
    }

    // use serial method for small inputs
    if(count < 64){
        return serial_find_extrema(first, last, sign, queue);
    }

    // use serial method for OpenCL version 1.0 due to
    // problems with atomic_cmpxchg()
    #ifndef CL_VERSION_1_1
        return serial_find_extrema(first, last, sign, queue);
    #endif

    return find_extrema_with_atomics(first, last, sign, queue);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_FIND_EXTREMA_HPP
