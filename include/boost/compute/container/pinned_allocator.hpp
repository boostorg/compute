//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_CONTAINER_PINNED_ALLOCATOR_HPP
#define BOOST_COMPUTE_CONTAINER_PINNED_ALLOCATOR_HPP

#include <boost/compute/container/allocator.hpp>

namespace boost {
namespace compute {

template<class T>
class pinned_allocator : public allocator<T>
{
public:
    pinned_allocator(const context &context)
        : allocator<T>(context)
    {
        allocator<T>::set_mem_flags(buffer::read_write | buffer::alloc_host_ptr);
    }

    pinned_allocator(const pinned_allocator<T> &other)
        : allocator<T>(other)
    {
    }

    pinned_allocator<T>& operator=(const pinned_allocator<T> &other)
    {
        if(this != &other){
            allocator<T>::operator=(other);
        }

        return *this;
    }

    ~pinned_allocator()
    {
    }
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_CONTAINER_PINNED_ALLOCATOR_HPP
