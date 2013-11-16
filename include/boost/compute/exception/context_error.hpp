//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_EXCEPTION_CONTEXT_ERROR_HPP
#define BOOST_COMPUTE_EXCEPTION_CONTEXT_ERROR_HPP

#include <exception>

namespace boost {
namespace compute {

class context;

class context_error : public std::exception
{
public:
    context_error(const context *context,
                  const char *errinfo,
                  const void *private_info,
                  size_t private_info_size) throw()
        : m_context(context),
          m_errinfo(errinfo),
          m_private_info(private_info),
          m_private_info_size(private_info_size)
    {
    }

    ~context_error() throw()
    {
    }

    const context* get_context() const throw()
    {
        return m_context;
    }

    const char* what() const throw()
    {
        return m_errinfo;
    }

private:
    const context *m_context;
    const char *m_errinfo;
    const void *m_private_info;
    size_t m_private_info_size;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_EXCEPTION_CONTEXT_ERROR_HPP
