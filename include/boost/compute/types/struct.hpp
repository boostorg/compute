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

#include <boost/static_assert.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/fold_left.hpp>
#include <boost/preprocessor/seq/transform.hpp>

#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/detail/variadic_macros.hpp>

namespace boost {
namespace compute {
namespace detail {

template<class Struct, class T>
inline std::string adapt_struct_insert_member(T Struct::*, const char *name)
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

/// \internal_
#define BOOST_COMPUTE_DETAIL_STRUCT_MEMBER_SIZE(s, struct_, member_) \
    sizeof(((struct_ *)0)->member_)

/// \internal_
#define BOOST_COMPUTE_DETAIL_STRUCT_MEMBER_SIZE_ADD(s, x, y) (x+y)

/// \internal_
#define BOOST_COMPUTE_DETAIL_STRUCT_MEMBER_SIZE_SUM(struct_, members_) \
    BOOST_PP_SEQ_FOLD_LEFT( \
        BOOST_COMPUTE_DETAIL_STRUCT_MEMBER_SIZE_ADD, \
        0, \
        BOOST_PP_SEQ_TRANSFORM( \
            BOOST_COMPUTE_DETAIL_STRUCT_MEMBER_SIZE, struct_, members_ \
        ) \
    )

/// \internal_
///
/// Returns true if struct_ contains no internal padding bytes (i.e. it is
/// packed). members_ is a sequence of the names of the struct members.
#define BOOST_COMPUTE_DETAIL_STRUCT_IS_PACKED(struct_, members_) \
    (sizeof(struct_) == BOOST_COMPUTE_DETAIL_STRUCT_MEMBER_SIZE_SUM(struct_, members_))

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
/// BOOST_COMPUTE_FUNCTION(bool, sort_by_x, (Particle a, Particle b),
/// {
///     return a.x < b.x;
/// });
///
/// // sort particles by their x-coordinate
/// boost::compute::sort(
///     particles.begin(), particles.end(), sort_by_x, queue
/// );
/// \endcode
///
/// Due to differences in struct padding between the host compiler and the
/// device compiler, the \c BOOST_COMPUTE_ADAPT_STRUCT() macro requires that
/// the adapted struct is packed (i.e. no padding bytes between members).
///
/// \see type_name()
#define BOOST_COMPUTE_ADAPT_STRUCT(type, name, members) \
    BOOST_STATIC_ASSERT_MSG( \
        BOOST_COMPUTE_DETAIL_STRUCT_IS_PACKED(type, BOOST_COMPUTE_PP_TUPLE_TO_SEQ(members)), \
        "BOOST_COMPUTE_ADAPT_STRUCT() does not support structs with internal padding." \
    ); \
    BOOST_COMPUTE_TYPE_NAME(type, name) \
    namespace boost { namespace compute { namespace detail { \
    template<> \
    struct inject_type_impl<type> \
    { \
        void operator()(meta_kernel &kernel) \
        { \
            std::stringstream declaration; \
            declaration << "typedef struct __attribute__((packed)) {\n" \
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
