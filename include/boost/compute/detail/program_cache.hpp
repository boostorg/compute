//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_PROGRAM_CACHE_HPP
#define BOOST_COMPUTE_DETAIL_PROGRAM_CACHE_HPP

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>

#include <boost/compute/context.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/detail/lru_cache.hpp>
#include <boost/compute/detail/global_static.hpp>

namespace boost {
namespace compute {
namespace detail {

class program_cache : boost::noncopyable
{
public:
    program_cache(size_t capacity)
        : m_cache(capacity)
    {
    }

    ~program_cache()
    {
    }

    size_t size() const
    {
        return m_cache.size();
    }

    size_t capacity() const
    {
        return m_cache.capacity();
    }

    void insert(const std::string &key, const program &program)
    {
        m_cache.insert(key, program);
    }

    program get(const std::string &key)
    {
        return m_cache.get(key);
    }

private:
    lru_cache<std::string, program> m_cache;
};

// returns the program cache for the context
inline boost::shared_ptr<program_cache> get_program_cache(const context &context)
{
    typedef lru_cache<cl_context, boost::shared_ptr<program_cache> > cache_map;

    BOOST_COMPUTE_DETAIL_GLOBAL_STATIC(cache_map, caches, (8));

    boost::shared_ptr<program_cache> cache = caches.get(context.get());
    if(!cache){
        cache = boost::make_shared<program_cache>(64);

        caches.insert(context.get(), cache);
    }

    return cache;
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_PROGRAM_CACHE_HPP
