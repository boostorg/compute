//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_RANDOM_UNIFORM_REAL_DISTRIBUTION_HPP
#define BOOST_COMPUTE_RANDOM_UNIFORM_REAL_DISTRIBUTION_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/meta_kernel.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class T, class Expr>
struct invoked_scale_random
{
    typedef T result_type;

    invoked_scale_random(const T &a, const T &b, const Expr &expr)
        : m_a(a),
          m_b(b),
          m_expr(expr)
    {
    }

    T m_a;
    T m_b;
    Expr m_expr;
};

template<class T, class Expr>
meta_kernel& operator<<(meta_kernel &kernel,
                        const invoked_scale_random<T, Expr> &expr)
{
    return kernel
        << expr.m_a << " + ("
        << "(convert_float(as_uint(" << expr.m_expr << ")) / UINT_MAX)"
        << "* ((" << expr.m_b << ")-(" << expr.m_a << ")))";
}

template<class T>
struct scale_random
{
    typedef T result_type;

    scale_random(T a, T b)
        : m_a(a),
          m_b(b)
    {
    }

    template<class Expr>
    invoked_scale_random<T, Expr> operator()(const Expr &expr) const
    {
        return invoked_scale_random<T, Expr>(m_a, m_b, expr);
    }

    T m_a;
    T m_b;
};

} // end detail namespace

/// \class uniform_real_distribution
/// \brief Produces uniformily distributed random floating-point numbers.
template<class RealType = float>
class uniform_real_distribution
{
public:
    typedef RealType result_type;

    /// Creates a new uniform distribution producing numbers in the range
    /// [\p a, \p b).
    uniform_real_distribution(RealType a = 0.f, RealType b = 1.f)
        : m_a(a),
          m_b(b)
    {
    }

    /// Destroys the uniform_real_distribution object.
    ~uniform_real_distribution()
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

    /// Generates uniformily distributed floating-point numbers and stores
    /// them to the range [\p first, \p last).
    template<class OutputIterator, class Generator>
    void generate(OutputIterator first,
                  OutputIterator last,
                  Generator &generator,
                  command_queue &queue)
    {
        generator.generate(
            first, last, detail::scale_random<RealType>(m_a, m_b), queue
        );
    }

    /// \internal_ (deprecated)
    template<class OutputIterator, class Generator>
    void fill(OutputIterator first,
              OutputIterator last,
              Generator &g,
              command_queue &queue)
    {
        generate(first, last, g, queue);
    }

private:
    RealType m_a;
    RealType m_b;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_RANDOM_UNIFORM_REAL_DISTRIBUTION_HPP
