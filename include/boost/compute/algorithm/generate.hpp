//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_GENERATE_HPP
#define BOOST_COMPUTE_ALGORITHM_GENERATE_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/iterator/function_input_iterator.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/detail/default_queue_for_iterator.hpp>

namespace boost {
namespace compute {

template<class OutputIterator, class Generator>
inline void generate(OutputIterator first,
                     OutputIterator last,
                     Generator generator,
                     command_queue &queue)
{
    size_t count = detail::iterator_range_size(first, last);
    if(count == 0){
        return;
    }

    ::boost::compute::copy(
        ::boost::compute::make_function_input_iterator(generator,
                                                       first.get_index()),
        ::boost::compute::make_function_input_iterator(generator,
                                                       last.get_index()),
        first,
        queue
    );
}

template<class OutputIterator, class Generator>
inline void generate(OutputIterator first,
                     OutputIterator last,
                     Generator generator)
{
    command_queue queue = detail::default_queue_for_iterator(first);

    ::boost::compute::generate(first, last, generator, queue);
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_GENERATE_HPP
