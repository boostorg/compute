//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_BLAS_ENUMS_HPP
#define BOOST_COMPUTE_BLAS_ENUMS_HPP

#include <boost/compute/cl.hpp>

namespace boost {
namespace compute {
namespace blas {

enum matrix_order {
    row_major,
    column_major
};

enum matrix_transpose {
    no_transpose,
    transpose,
    conjugate_transpose
};

} // end blas namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_BLAS_ENUMS_HPP
