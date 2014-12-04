//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_EXTENTS_HPP
#define BOOST_COMPUTE_EXTENTS_HPP

#include <functional>
#include <numeric>

#include <boost/compute/config.hpp>

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && \
    !defined(BOOST_NO_0X_HDR_INITIALIZER_LIST)
#include <initializer_list>
#endif

#include <boost/array.hpp>

namespace boost {
namespace compute {

/// The extents class contains an array of n-dimensional extents.
///
/// \see dim()
template<size_t N>
class extents
{
public:
    typedef size_t size_type;
    static const size_type static_size = N;

    /// Creates an extents object with each component set to zero.
    ///
    /// For example:
    /// \code
    /// extents<3> exts(); // (0, 0, 0)
    /// \endcode
    extents()
    {
        m_extents.fill(0);
    }

    /// Creates an extents object with each component set to \p value.
    ///
    /// For example:
    /// \code
    /// extents<3> exts(1); // (1, 1, 1)
    /// \endcode
    explicit extents(size_t value)
    {
        m_extents.fill(value);
    }

    #if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && \
        !defined(BOOST_NO_0X_HDR_INITIALIZER_LIST)
    /// Creates an extents object with \p values.
    extents(std::initializer_list<size_t> values)
    {
        BOOST_ASSERT(values.size() == N);

        std::copy(values.begin(), values.end(), m_extents.begin());
    }
    #endif

    /// Returns the size (i.e. dimensionality) of the extents array.
    size_type size() const
    {
        return N;
    }

    /// Returns the linear size of the extents. This is equivalent to the
    /// product of each extent in each dimension.
    size_type linear() const
    {
        return std::accumulate(
            m_extents.begin(), m_extents.end(), 1, std::multiplies<size_type>()
        );
    }

    /// Returns a pointer to the extents data array.
    ///
    /// This is useful for passing the extents data to OpenCL APIs which
    /// expect an array of \c size_t.
    size_t* data()
    {
        return m_extents.data();
    }

    /// \overload
    const size_t* data() const
    {
        return m_extents.data();
    }

    /// Returns a reference to the extent at \p index.
    size_t& operator[](size_t index)
    {
        return m_extents[index];
    }

    /// \overload
    const size_t& operator[](size_t index) const
    {
        return m_extents[index];
    }

    /// Returns \c true if the extents in \c *this are the same as \p other.
    bool operator==(const extents &other) const
    {
        return m_extents == other.m_extents;
    }

    /// Returns \c true if the extents in \c *this are not the same as \p other.
    bool operator!=(const extents &other) const
    {
        return m_extents != other.m_extents;
    }

private:
    boost::array<size_t, N> m_extents;
};

#ifndef BOOST_NO_VARIADIC_TEMPLATES
/// The variadic dim() function provides a concise syntax for creating
/// \ref extents objects.
///
/// For example,
/// \code
/// extents<2> region = dim(640, 480); // region == (640, 480)
/// \endcode
///
/// \see extents<N>
template<class... Args>
inline extents<sizeof...(Args)> dim(Args... args)
{
    return extents<sizeof...(Args)>({ static_cast<size_t>(args)... });
}
#else
// dim() function definitions for non-c++11 compilers
#define BOOST_COMPUTE_DETAIL_ASSIGN_DIM(z, n, var) \
    var[n] = BOOST_PP_CAT(e, n);

#define BOOST_COMPUTE_DETAIL_DEFINE_DIM(z, n, var) \
    inline extents<n> dim(BOOST_PP_ENUM_PARAMS(n, size_t e)) \
    { \
        extents<n> exts; \
        BOOST_PP_REPEAT(n, BOOST_COMPUTE_DETAIL_ASSIGN_DIM, exts) \
        return exts; \
    }

BOOST_PP_REPEAT(BOOST_COMPUTE_MAX_ARITY, BOOST_COMPUTE_DETAIL_DEFINE_DIM, ~)

#undef BOOST_COMPUTE_DETAIL_ASSIGN_DIM
#undef BOOST_COMPUTE_DETAIL_DEFINE_DIM

#endif // BOOST_NO_VARIADIC_TEMPLATES

/// \internal_
template<size_t N>
inline extents<N> dim()
{
    return extents<N>();
}

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_EXTENTS_HPP
