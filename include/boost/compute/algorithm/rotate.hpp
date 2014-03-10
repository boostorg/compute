#ifndef BOOST_COMPUTE_ALGORITHM_ROTATE_HPP
#define BOOST_COMPUTE_ALGORITHM_ROTATE_HPP

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/copy.hpp>

namespace boost {
namespace compute {

/// Performs left rotation such that element at
/// n_first comes to the beginning
template<class InputIterator>
void rotate(InputIterator first, 
			InputIterator n_first, 
			InputIterator last,
			command_queue &queue = system::default_queue())
{
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    size_t count = detail::iterator_range_size(first, n_first);

    vector<T> temp(count);
    ::boost::compute::copy(first,n_first,temp.begin(),queue);

    ::boost::compute::copy(n_first,last,first,queue);
    ::boost::compute::copy(temp.begin(),temp.end(),last-count,queue);
}

} //end compute namespace
} //end boost namespace

#endif // BOOST_COMPUTE_ALGORITHM_ROTATE_HPP