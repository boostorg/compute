//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_FIXED_SORT_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_FIXED_SORT_HPP

#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/program_cache.hpp>

namespace boost {
namespace compute {
namespace detail {

// sort two values
template<class T>
inline void sort2(const buffer &buffer, command_queue &queue)
{
    const context &context = queue.get_context();

    boost::shared_ptr<detail::program_cache> cache =
        detail::get_program_cache(context);
    std::string cache_key =
        std::string("fixed_sort2_") + type_name<T>();

    program sort2_program = cache->get(cache_key);
    if(!sort2_program.get()){
        const char source[] =
            "__kernel void sort2(__global T *input)\n"
            "{\n"
            "    const T x = input[0];\n"
            "    const T y = input[1];\n"
            "    if(y < x){\n"
            "        input[0] = y;\n"
            "        input[1] = x;\n"
            "    }\n"
            "}\n";

        sort2_program = program::create_with_source(source, context);
        sort2_program.build(std::string("-DT=") + type_name<T>());

        cache->insert(cache_key, sort2_program);
    }

    kernel sort2_kernel = sort2_program.create_kernel("sort2");
    sort2_kernel.set_arg(0, buffer);
    queue.enqueue_task(sort2_kernel);
}

// sort three values
template<class T>
inline void sort3(const buffer &buffer, command_queue &queue)
{
    const context &context = queue.get_context();

    boost::shared_ptr<detail::program_cache> cache =
        detail::get_program_cache(context);
    std::string cache_key =
        std::string("fixed_sort3_") + type_name<T>();

    program sort3_program = cache->get(cache_key);
    if(!sort3_program.get()){
        const char source[] =
            "__kernel void sort3(__global T *input)\n"
            "{\n"
            "    const T x = input[0];\n"
            "    const T y = input[1];\n"
            "    const T z = input[2];\n"
            "    if(y < x){\n"
            "         if(z < x){\n"
            "             if(z < y){\n"
            "                 input[0] = z;\n"
            "                 input[1] = y;\n"
            "                 input[2] = x;\n"
            "             }\n"
            "             else {\n"
            "                 input[0] = y;\n"
            "                 input[1] = z;\n"
            "                 input[2] = x;\n"
            "             }\n"
            "         }\n"
            "         else {\n"
            "            input[0] = y;\n"
            "            input[1] = x;\n"
            "         }\n"
            "    }\n"
            "    else {\n"
            "        if(z < x){\n"
            "            input[0] = z;\n"
            "            input[1] = x;\n"
            "            input[2] = y;\n"
            "        }\n"
            "        else if(z < y){\n"
            "            input[1] = z;\n"
            "            input[2] = y;\n"
            "        }\n"
            "    }\n"
            "}\n";

        sort3_program = program::create_with_source(source, context);
        sort3_program.build(std::string("-DT=") + type_name<T>());

        cache->insert(cache_key, sort3_program);
    }

    kernel sort3_kernel = sort3_program.create_kernel("sort3");
    sort3_kernel.set_arg(0, buffer);
    queue.enqueue_task(sort3_kernel);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_FIXED_SORT_HPP
