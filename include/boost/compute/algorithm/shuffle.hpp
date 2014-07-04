//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Mageswaran.D <mageswaran1989@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_SHUFFLE_HPP
#define BOOST_COMPUTE_ALGORITHM_SHUFFLE_HPP

#include <vector>
#include <algorithm>

#include <boost/compute/system.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/random/uniform_int_distribution.hpp>


namespace boost {
namespace compute {
namespace detail {

const char shuffle_source[] =
"__kernel void gpu_shuffle(__global T *in_data,\n"
"                          __global const unsigned int *rand_data)\n"
"{\n"
"   int i = get_global_id(0);\n"
"   int shuffle_index = rand_data[i];\n"
"   float temp = in_data[shuffle_index];\n"
"   barrier(CLK_GLOBAL_MEM_FENCE);\n"
"   in_data[i] = temp;\n"
"   }\n";


template<class InputIterator, class Generator>
inline void shuffle_impl(InputIterator first,
                    InputIterator last,
                    Generator &rng,
                    command_queue &queue = system::default_queue())
{
    const size_t size = detail::iterator_range_size(first, last);
    typedef typename std::iterator_traits<InputIterator>::value_type suffle_type;
    if(size == 0){
        return;
    }
    compute::vector<unsigned int> gpu_random_index_vec(size);

    compute::uniform_int_distribution<unsigned int> random_index_generator(0, size);
    random_index_generator.generate(gpu_random_index_vec.begin(),
                                    gpu_random_index_vec.end(),
                                    rng,
                                    queue);

    program shuffle_program = program::create_with_source(shuffle_source,
                                                          queue.get_context());
    std::stringstream options;
    options << " -DT=" << type_name<suffle_type>();
    shuffle_program.build(options.str());

    kernel shuffle_kernel(shuffle_program, "gpu_shuffle");
    shuffle_kernel.set_arg(0, first.get_buffer());
    shuffle_kernel.set_arg(1, gpu_random_index_vec);
    queue.enqueue_nd_range_kernel(shuffle_kernel,
                                  1,
                                  0,
                                  &size,
                                  0);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

namespace boost {
namespace compute {

/// Reorders the elements in the given range [first, last) such that
/// each possible permutation of those elements has equal probability
/// of appearance.

template<class Iterator, class Generator>
inline void shuffle(Iterator first,
                    Iterator last,
                    Generator &rng,
                    command_queue &queue = system::default_queue())
{
    boost::compute::detail::shuffle_impl(first, last, rng, queue);
}

} // end compute namespace
} // end boost namespace
#endif // SHUFFLE_HPP
