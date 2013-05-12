//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_COUNT_IF_WITH_ATOMICS_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_COUNT_IF_WITH_ATOMICS_HPP

#include <iterator>

#include <boost/compute/container/detail/scalar.hpp>
#include <boost/compute/functional/atomic.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class Predicate>
class count_if_with_atomics_kernel : meta_kernel
{
public:
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    count_if_with_atomics_kernel()
        : meta_kernel("count_if_with_atomics")
    {
        m_count = 0;

        m_count_arg_index = add_arg<uint_ *>("__global", "count");
    }

    void set_args(InputIterator first,
                  InputIterator last,
                  Predicate predicate)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        m_count = detail::iterator_range_size(first, last);

        atomic_inc<uint_> atomic_inc_uint;

        *this << decl<const T>("value") << "=" << first[get_global_id(0)] << ";\n"
              << if_(predicate(var<const T>("value"))) << "{\n"
              << "    " << atomic_inc_uint(expr<uint_>("count")) << ";\n"
              << "}\n";
    }

    size_t exec(command_queue &queue)
    {
        const context &context = queue.get_context();

        // setup count buffer
        scalar<uint_> count(context);
        set_arg(m_count_arg_index, count.get_buffer());

        // initialize count to zero
        count.write(0, queue);

        // execute kernel
        exec_1d(queue, 0, m_count);

        // read and return count
        return count.read(queue);
    }

private:
    size_t m_count;
    size_t m_count_arg_index;
};

// counts the number of elements that match the predicate using atomic_inc()
template<class InputIterator, class Predicate>
inline size_t count_if_with_atomics(InputIterator first,
                                    InputIterator last,
                                    Predicate predicate,
                                    command_queue &queue)
{
    count_if_with_atomics_kernel<InputIterator, Predicate> kernel;
    kernel.set_args(first, last, predicate);
    return kernel.exec(queue);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_COUNT_IF_WITH_ATOMICS_HPP
