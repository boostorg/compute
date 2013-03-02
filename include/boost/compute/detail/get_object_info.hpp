//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_DETAIL_GET_OBJECT_INFO_HPP
#define BOOST_COMPUTE_DETAIL_GET_OBJECT_INFO_HPP

#include <string>

#include <boost/throw_exception.hpp>

#include <boost/compute/exception.hpp>

namespace boost {
namespace compute {
namespace detail {

// default implementation
template<class T, class Function, class Object, class Info>
struct _get_object_info_impl
{
    T operator()(Function function, Object object, Info info)
    {
        T value;
        cl_int ret = function(object,
                              static_cast<cl_uint>(info),
                              sizeof(T),
                              &value,
                              0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return value;
    }
};

// specialization for bool
template<class Function, class Object, class Info>
struct _get_object_info_impl<bool, Function, Object, Info>
{
    bool operator()(Function function, Object object, Info info)
    {
        cl_bool value;
        cl_int ret = function(object,
                              static_cast<cl_uint>(info),
                              sizeof(cl_bool),
                              &value,
                              0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return value == CL_TRUE;
    }
};

// specialization for std::string
template<class Function, class Object, class Info>
struct _get_object_info_impl<std::string, Function, Object, Info>
{
    std::string operator()(Function function, Object object, Info info)
    {
        size_t size = 0;
        cl_int ret = function(object,
                              static_cast<cl_uint>(info),
                              0,
                              0,
                              &size);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        std::string value(size - 1, 0);
        ret = function(object,
                       static_cast<cl_uint>(info),
                       size,
                       &value[0],
                       0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return value;
    }
};

// returns the value (of type T) from the given clGet*Info() function call.
template<class T, class Function, class Object, class Info>
T get_object_info(Function f, Object o, Info i)
{
    return _get_object_info_impl<T, Function, Object, Info>()(f, o, i);
}

} // end detail namespace
} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_DETAIL_GET_OBJECT_INFO_HPP
