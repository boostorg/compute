//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_EXCEPTION_EXTENSION_UNSUPPORTED_EXCEPTION_HPP
#define BOOST_COMPUTE_EXCEPTION_EXTENSION_UNSUPPORTED_EXCEPTION_HPP

#include <exception>
#include <sstream>
#include <string>

namespace boost {
namespace compute {

/// \class extension_unsupported_exception
/// \brief Exception thrown when attempting to use an unsupported
///        OpenCL extension.
///
/// This exception is thrown when the user attempts to use an OpenCL
/// extension which is not supported on the platform and/or device.
///
/// An example of this is attempting to use CL-GL sharing on a non-GPU
/// device.
///
/// \see runtime_exception
class extension_unsupported_exception : public std::exception
{
public:
    explicit extension_unsupported_exception(const char *extension) throw()
    {
        std::stringstream msg;
        msg << "OpenCL extension " << extension << " not supported";
        m_error_string = msg.str();
    }

    ~extension_unsupported_exception() throw()
    {
    }

    const char* what() const throw()
    {
        return m_error_string.c_str();
    }

private:
    std::string m_error_string;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_EXCEPTION_EXTENSION_UNSUPPORTED_EXCEPTION_HPP
