//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_TYPES_STRUCT_HPP
#define BOOST_COMPUTE_TYPES_STRUCT_HPP

#include <sstream>

#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/variadic_macros.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class Struct, class T>
inline std::string adapt_struct_insert_member(T Struct::* member, const char *name)
{
    std::stringstream s;
    s << "    " << type_name<T>() << " " << name << ";\n";
    return s.str();
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

/// \internal_
#define BOOST_COMPUTE_DETAIL_ADAPT_STRUCT_INSERT_MEMBER(r, type, member) \
    << ::boost::compute::detail::adapt_struct_insert_member( \
           &type::member, BOOST_PP_STRINGIZE(member) \
       )

/// The BOOST_COMPUTE_ADAPT_STRUCT() macro makes a C++ struct/class available
/// to OpenCL kernels.
///
/// \param type The C++ type.
/// \param name The OpenCL name.
/// \param members A tuple of the struct's members.
///
/// For example, to adapt a 2D particle struct with position (x, y) and
/// velocity (dx, dy):
/// \code
/// // c++ struct definition
/// struct Particle
/// {
///     float x, y;
///     float dx, dy;
/// };
///
/// // adapt struct for OpenCL
/// BOOST_COMPUTE_ADAPT_STRUCT(Particle, Particle, (x, y, dx, dy))
/// \endcode
///
/// After adapting the struct it can be used in Boost.Compute containers
/// and with Boost.Compute algorithms:
/// \code
/// // create vector of particles
/// boost::compute::vector<Particle> particles = ...
///
/// // function to compare particles by their x-coordinate
/// BOOST_COMPUTE_FUNCTION(bool, sort_by_x, (Particle, Particle),
/// {
///     return _1.x < _2.x;
/// });
///
/// // sort particles by their x-coordinate
/// boost::compute::sort(
///     particles.begin(), particles.end(), sort_by_x, queue
/// );
/// \endcode
#define BOOST_COMPUTE_ADAPT_STRUCT(type, name, members) \
    BOOST_COMPUTE_TYPE_NAME(type, name) \
    namespace boost { namespace compute { namespace detail { \
    template<> \
    struct inject_type_impl<type> \
    { \
        void operator()(meta_kernel &kernel) \
        { \
            std::stringstream declaration; \
            declaration << "typedef struct {\n" \
                        BOOST_PP_SEQ_FOR_EACH( \
                            BOOST_COMPUTE_DETAIL_ADAPT_STRUCT_INSERT_MEMBER, \
                            type, \
                            BOOST_COMPUTE_PP_TUPLE_TO_SEQ(members) \
                        ) \
                        << "} " << type_name<type>() << ";\n"; \
            kernel.add_type_declaration<type>(declaration.str()); \
        } \
    }; \
    }}}

#endif // BOOST_COMPUTE_TYPES_STRUCT_HPP
