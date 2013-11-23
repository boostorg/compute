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

#include <boost/compute/future.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace boost {
namespace compute {
namespace detail {

inline size_t pick_copy_work_group_size(size_t n, const device &device)
{
    (void) device;

    if(n % 32 == 0) return 32;
    else if(n % 16 == 0) return 16;
    else if(n % 8 == 0) return 8;
    else if(n % 4 == 0) return 4;
    else if(n % 2 == 0) return 2;
    else return 1;
}

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
        m_count_arg = add_arg<uint_>("count");

        *this <<
            "const uint i = get_global_id(0);\n" <<
            "if(i < count){\n" <<
                result[expr<uint_>("i")] << '='
                    << first[expr<uint_>("i")] << ";\n" <<
            "}\n";

        m_count = detail::iterator_range_size(first, last);
    }

    event exec(command_queue &queue)
    {
        if(m_count == 0){
            // nothing to do
            return event();
        }

        size_t work_group_size = 256;
        size_t global_work_size = m_count;

        if(global_work_size % work_group_size != 0){
            global_work_size +=
                work_group_size - global_work_size % work_group_size;
        }

        set_arg(m_count_arg, uint_(m_count));

        return exec_1d(queue, 0, global_work_size, work_group_size);
    }

private:
    size_t m_count;
    size_t m_count_arg;
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

template<class InputIterator, class OutputIterator>
inline future<OutputIterator> copy_on_device_async(InputIterator first,
                                                   InputIterator last,
                                                   OutputIterator result,
                                                   command_queue &queue)
{
    copy_kernel<InputIterator, OutputIterator> kernel;

    kernel.set_range(first, last, result);
    event event_ = kernel.exec(queue);

    return make_future(result + std::distance(first, last), event_);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_ON_DEVICE_HPP
