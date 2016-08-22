//---------------------------------------------------------------------------//
// Copyright (c) 2016 Jakub Szuppe <j.szuppe@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_WEIGHT_FUNC_HPP
#define BOOST_COMPUTE_DETAIL_WEIGHT_FUNC_HPP

#include <vector>

#include <boost/compute/distributed/context.hpp>
#include <boost/compute/distributed/command_queue.hpp>

namespace boost {
namespace compute {
namespace distributed {

typedef std::vector<double> (*weight_func)(const command_queue&);

namespace detail {

/// \internal_
/// Rounds up \p n to the nearest multiple of \p m.
/// Note: \p m must be a multiple of 2.
size_t round_up(size_t n, size_t m)
{
    assert(m && ((m & (m -1)) == 0));
    return (n + m - 1) & ~(m - 1);
}

/// \internal_
///
std::vector<size_t> partition(const command_queue& queue,
                              weight_func weight_func,
                              const size_t size,
                              const size_t align)
{
    std::vector<double> weights = weight_func(queue);
    std::vector<size_t> partition;
    partition.reserve(queue.size() + 1);
    partition.push_back(0);

    if(queue.size() > 1)
    {
        double acc = 0;
        for(size_t i = 0; i < queue.size(); i++)
        {
            acc += weights[i];
            partition.push_back(
                std::min(
                    size,
                    round_up(size * acc, align)
                )
            );
        }
        return partition;
    }
    partition.push_back(size);
    return partition;
}

} // end distributed detail

std::vector<double> default_weight_func(const command_queue& queue)
{
    return std::vector<double>(queue.size(), 1.0/queue.size());
}

} // end distributed namespace
} // end compute namespace
} // end boost namespace


#endif /* INCLUDE_BOOST_COMPUTE_DETAIL_WEIGHT_FUNC_HPP_ */
