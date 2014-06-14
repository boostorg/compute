//---------------------------------------------------------------------------//
// Copyright (c) 2014 Roshan <thisisroshansmail@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP
#define BOOST_COMPUTE_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/types/builtin.hpp>
#include <boost/compute/algorithm/copy_if.hpp>
#include <boost/compute/algorithm/transform.hpp>

namespace boost {
namespace compute {

/// \class uniform_int_distribution
/// \brief Produces uniformily distributed random integers
///
/// The following example shows how to setup a uniform int distribution to
/// produce random integers 0 and 1.
///
/// \snippet test/test_uniform_int_distribution.cpp generate
///
template<class IntType = uint_>
class uniform_int_distribution
{
public:
    typedef IntType result_type;

    /// Creates a new uniform distribution producing numbers in the range
    /// [\p a, \p b].
    uniform_int_distribution(IntType a = 0, IntType b = 1)
        : m_a(a),
          m_b(b)
    {
    }

    /// Destroys the uniform_int_distribution object.
    ~uniform_int_distribution()
    {
    }

    /// Returns the minimum value of the distribution.
    result_type a() const
    {
        return m_a;
    }

    /// Returns the maximum value of the distribution.
    result_type b() const
    {
        return m_b;
    }

    /// Generates uniformily distributed integers and stores
    /// them to the range [\p first, \p last).
    template<class OutputIterator, class Generator>
    void generate(OutputIterator first,
                  OutputIterator last,
                  Generator &generator,
                  command_queue &queue)
    {
        size_t size = std::distance(first, last);
        vector<IntType> tmp(size, queue.get_context());

        uint_ bound = ((uint(-1))/(m_b-m_a+1))*(m_b-m_a+1);

        OutputIterator new_first = first;

        while(size>0)
        {
            generator.generate(tmp.begin(), tmp.begin() + size, queue);
            new_first = copy_if(tmp.begin(), tmp.begin() + size, first,
                                _1 <= bound, queue);
            size = std::distance(new_first, last);
        }

        BOOST_COMPUTE_FUNCTION(IntType, scale_random, (const IntType x),
        {
            return LO + (x % (HI-LO+1));
        });

        scale_random.define("LO", boost::lexical_cast<std::string>(m_a));
        scale_random.define("HI", boost::lexical_cast<std::string>(m_b));

        transform(first, last, first, scale_random, queue);
    }

private:
    IntType m_a;
    IntType m_b;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP
