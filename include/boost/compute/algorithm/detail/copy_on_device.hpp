//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_ON_DEVICE_HPP
#define BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_ON_DEVICE_HPP

#include <iterator>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/async/future.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
#include <boost/compute/iterator/discard_iterator.hpp>
#include <boost/compute/memory/svm_ptr.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/parameter_cache.hpp>
#include <boost/compute/detail/work_size.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class InputIterator, class OutputIterator>
inline event copy_on_device_cpu(InputIterator first,
                                OutputIterator result,
                                size_t count,
                                command_queue &queue)
{
    meta_kernel k("copy");
    const device& device = queue.get_device();

    k <<
        "uint block = " <<
            "(uint)ceil(((float)count)/get_global_size(0));\n" <<
        "uint start = get_global_id(0) * block;\n" <<
        "uint end = min(count, start + block);\n" <<
        "for(uint i = start; i < end; i++){\n" <<
            result[k.var<uint_>("i")] << '=' <<
                first[k.var<uint_>("i")] << ";\n" <<
        "}\n";

    k.add_set_arg<const uint_>("count", count);

    size_t global_work_size =
        device.compute_units();
    if(count <= 1024) global_work_size = 1;
    return k.exec_1d(queue, 0, global_work_size);
}

template<class InputIterator, class OutputIterator>
inline event copy_on_device_gpu(InputIterator first,
                                OutputIterator result,
                                size_t count,
                                command_queue &queue)
{
    typedef typename std::iterator_traits<InputIterator>::value_type input_type;

    const device& device = queue.get_device();
    boost::shared_ptr<parameter_cache> parameters =
        detail::parameter_cache::get_global_cache(device);
    std::string cache_key =
        "__boost_copy_kernel_" + boost::lexical_cast<std::string>(sizeof(input_type));

    uint_ vpt = parameters->get(cache_key, "vpt", 4);
    uint_ tpb = parameters->get(cache_key, "tpb", 128);

    meta_kernel k("copy");
    k <<
        "uint index = get_local_id(0) + " <<
            "(" << vpt * tpb << " * get_group_id(0));\n" <<
        "for(uint i = 0; i < " << vpt << "; i++){\n" <<
        "    if(index < count){\n" <<
                result[k.var<uint_>("index")] << '=' <<
                    first[k.var<uint_>("index")] << ";\n" <<
        "       index += " << tpb << ";\n"
        "    }\n"
        "}\n";

    k.add_set_arg<const uint_>("count", count);
    size_t global_work_size = calculate_work_size(count, vpt, tpb);
    return k.exec_1d(queue, 0, global_work_size, tpb);
}

template<class InputIterator, class OutputIterator>
inline event dispatch_copy_on_device(InputIterator first,
                                     InputIterator last,
                                     OutputIterator result,
                                     command_queue &queue)
{
    const size_t count = detail::iterator_range_size(first, last);

    if(count == 0){
        // nothing to do
        return event();
    }

    const device& device = queue.get_device();
    if(device.type() & device::cpu) {
        return copy_on_device_cpu(first, result, count, queue);
    }
    return copy_on_device_gpu(first, result, count, queue);
}

template<class InputIterator, class OutputIterator>
inline OutputIterator copy_on_device(InputIterator first,
                                     InputIterator last,
                                     OutputIterator result,
                                     command_queue &queue)
{
    dispatch_copy_on_device(first, last, result, queue);
    return result + std::distance(first, last);
}

template<class InputIterator>
inline discard_iterator copy_on_device(InputIterator first,
                                       InputIterator last,
                                       discard_iterator result,
                                       command_queue &queue)
{
    (void) queue;

    return result + std::distance(first, last);
}

template<class InputIterator, class OutputIterator>
inline future<OutputIterator> copy_on_device_async(InputIterator first,
                                                   InputIterator last,
                                                   OutputIterator result,
                                                   command_queue &queue)
{
    event event_ = dispatch_copy_on_device(first, last, result, queue);
    return make_future(result + std::distance(first, last), event_);
}

#ifdef CL_VERSION_2_0
// copy_on_device() specialization for svm_ptr
template<class T>
inline svm_ptr<T> copy_on_device(svm_ptr<T> first,
                                 svm_ptr<T> last,
                                 svm_ptr<T> result,
                                 command_queue &queue)
{
    size_t count = iterator_range_size(first, last);
    if(count == 0){
        return result;
    }

    queue.enqueue_svm_memcpy(
        result.get(), first.get(), count * sizeof(T)
    );

    return result + count;
}

template<class T>
inline future<svm_ptr<T> > copy_on_device_async(svm_ptr<T> first,
                                                svm_ptr<T> last,
                                                svm_ptr<T> result,
                                                command_queue &queue)
{
    size_t count = iterator_range_size(first, last);
    if(count == 0){
        return future<svm_ptr<T> >();
    }

    event event_ = queue.enqueue_svm_memcpy_async(
        result.get(), first.get(), count * sizeof(T)
    );

    return make_future(result + count, event_);
}
#endif // CL_VERSION_2_0

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_DETAIL_COPY_ON_DEVICE_HPP
