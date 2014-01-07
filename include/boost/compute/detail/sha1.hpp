//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_SHA1_HPP
#define BOOST_COMPUTE_DETAIL_SHA1_HPP

#include <sstream>
#include <iomanip>
#include <boost/uuid/sha1.hpp>

namespace boost {
namespace compute {
namespace detail {

// Returns SHA1 hash of the string parameter.
inline std::string sha1(const std::string &src) {
    boost::uuids::detail::sha1 sha1;
    sha1.process_bytes(src.c_str(), src.size());

    unsigned int hash[5];
    sha1.get_digest(hash);

    std::ostringstream buf;
    for(int i = 0; i < 5; ++i)
        buf << std::hex << std::setfill('0') << std::setw(8) << hash[i];

    return buf.str();
}

} // end detail namespace
} // end compute namespace
} // end boost namespace


#endif // BOOST_COMPUTE_DETAIL_SHA1_HPP
