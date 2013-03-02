//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_ON_DEVICE_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_ON_DEVICE_HPP

#include <iterator>

#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class OutputIterator>
class copy_kernel : public meta_kernel
{
public:
    copy_kernel()
        : meta_kernel("copy")
    {
        m_count = 0;
    }

    void set_range(InputIterator first,
                   InputIterator last,
                   OutputIterator result)
    {
        *this <<
            "const uint i = get_global_id(0);\n" <<
            result[expr<uint_>("i")] << '='
                << first[expr<uint_>("i")] << ";\n";

        m_count = detail::iterator_range_size(first, last);
    }

    void exec(command_queue &queue)
    {
        if(m_count == 0){
            // nothing to do
            return;
        }

        exec_1d(queue, 0, m_count);
    }

private:
    size_t m_count;
};

template<class InputIterator, class OutputIterator>
inline OutputIterator copy_on_device(InputIterator first,
                                     InputIterator last,
                                     OutputIterator result,
                                     command_queue &queue)
{
    copy_kernel<InputIterator, OutputIterator> kernel;

    kernel.set_range(first, last, result);
    kernel.exec(queue);

    return result + std::distance(first, last);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_ON_DEVICE_HPP
