//---------------------------------------------------------------------------//
// Copyright (c) 2014 Roshan <thisisroshansmail@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_BINARY_FIND_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_BINARY_FIND_HPP

#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/find_if.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/parameter_cache.hpp>

namespace boost {
namespace compute {
namespace detail{

///
/// \brief Binary find kernel class
///
/// Subclass of meta_kernel to perform single step in binary find.
///
class binary_find_kernel : public meta_kernel
{
public:
    binary_find_kernel(size_t threads) : meta_kernel("binary_find")
    {
        m_threads = threads;
    }

    template<class InputIterator, class UnaryPredicate>
    void set_range(InputIterator first,
                   InputIterator last,
                   UnaryPredicate predicate)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type value_type;
        int block = (iterator_range_size(first, last)-1)/(m_threads-1);

        m_index_arg = add_arg<uint_ *>(memory_object::global_memory, "index");

        atomic_min<uint_> atomic_min_uint;

        *this <<
            "uint i = get_global_id(0) * " << block << ";\n" <<
            decl<value_type>("value") << "=" << first[var<uint_>("i")] << ";\n" <<
            "if(" << predicate(var<value_type>("value")) << ") {\n" <<
                atomic_min_uint(var<uint_ *>("index"), var<uint_>("i")) << ";\n" <<
            "}\n";

    }

    event exec(command_queue &queue, scalar<uint_> index)
    {
        set_arg(m_index_arg, index.get_buffer());

        return exec_1d(queue, 0, m_threads);
    }

private:
    size_t m_threads;
    size_t m_index_arg;
};

///
/// \brief Binary find algorithm
///
/// Finds the end of true values in the partitioned range [first, last).
/// \return Iterator pointing to end of true values
///
/// \param first Iterator pointing to start of range
/// \param last Iterator pointing to end of range
/// \param predicate Predicate according to which the range is partitioned
/// \param queue Queue on which to execute
///
template<class InputIterator, class UnaryPredicate>
inline InputIterator binary_find(InputIterator first,
                                 InputIterator last,
                                 UnaryPredicate predicate,
                                 command_queue &queue = system::default_queue())
{
    const device &device = queue.get_device();

    boost::shared_ptr<parameter_cache> parameters =
        detail::parameter_cache::get_global_cache(device);

    const std::string cache_key = "__boost_binary_find";

    size_t find_if_limit = 128;
    size_t threads = parameters->get(cache_key, "tpb", 128);
    size_t count = iterator_range_size(first, last);

    InputIterator search_first = first;
    InputIterator search_last = last;

    while(count > find_if_limit) {

        scalar<uint_> index(queue.get_context());
        index.write(static_cast<uint_>(count), queue);

        binary_find_kernel kernel(threads);
        kernel.set_range(search_first, search_last, predicate);
        kernel.exec(queue, index);

        size_t i = index.read(queue);

        if(i == count) {
            search_first = search_last - ((count - 1)%(threads - 1));
            break;
        } else {
            search_last = search_first + i;
            search_first = search_last - ((count - 1)/(threads - 1));
        }

        // Make sure that first and last stay within the input range
        search_last = std::min(search_last, last);
        search_last = std::max(search_last, first);

        search_first = std::max(search_first, first);
        search_first = std::min(search_first, last);

        count = iterator_range_size(search_first, search_last);
    }

    return find_if(search_first, search_last, predicate, queue);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_BINARY_FIND_HPP
