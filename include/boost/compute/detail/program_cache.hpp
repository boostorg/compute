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

#include <map>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>

#include <boost/compute/context.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/detail/lru_cache.hpp>

namespace boost {
namespace compute {
namespace detail {

class program_cache : boost::noncopyable
{
public:
    program_cache()
        : m_cache(64)
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
    typedef std::map<cl_context, boost::shared_ptr<program_cache> > cache_map;

    static cache_map caches;

    cache_map::iterator i = caches.find(context.get());
    if(i != caches.end()){
        return i->second;
    }
    else {
        boost::shared_ptr<program_cache> cache = boost::make_shared<program_cache>();
        caches[context.get()] = cache;
        return cache;
    }
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_PROGRAM_CACHE_HPP
