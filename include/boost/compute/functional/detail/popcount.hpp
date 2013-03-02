//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_FUNCTIONAL_DETAIL_POPCOUNT_HPP
#define BOOST_COMPUTE_FUNCTIONAL_DETAIL_POPCOUNT_HPP

#include <boost/compute/function.hpp>

namespace boost {
namespace compute {
namespace detail {

// builtin popcount() is only available if the OpenCL version is 1.2 or later
template<class T>
class builtin_popcount : public function<T(T)>
{
public:
    builtin_popcount()
        : function<T(T)>("popcount")
    {
    }
};

// generic fallback implementation of popcount()
template<class T>
class generic_popcount : public function<T(T)>
{
public:
    generic_popcount()
        : function<T(T)>("generic_popcount")
    {
        this->set_source(
            "inline uint generic_popcount(const uint x)\n"
            "{\n"
            "  uint count = 0;\n"
            "  for(uint i = 0; i < sizeof(uint) * CHAR_BIT; i++){\n"
            "    if(x & 1 << i)\n"
            "        count++;\n"
            "  }\n"
            "  return count;\n"
            "}\n"
        );
    }
};

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_FUNCTIONAL_DETAIL_POPCOUNT_HPP
