//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_EXCEPTION_RUNTIME_EXCEPTION_HPP
#define BOOST_COMPUTE_EXCEPTION_RUNTIME_EXCEPTION_HPP

#include <exception>

#include <boost/compute/cl.hpp>

namespace boost {
namespace compute {

class runtime_exception : public std::exception
{
public:
    explicit runtime_exception(cl_int error) throw()
        : m_error(error)
    {
    }

    ~runtime_exception() throw()
    {
    }

    cl_int get_error() const throw()
    {
        return m_error;
    }

    const char* what() const throw()
    {
        switch(m_error){
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device Not Found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device Not Available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler Not Available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory Object Allocation Failure";
        case CL_OUT_OF_RESOURCES: return "Out of Resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of Host Memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling Information Not Available";
        case CL_MEM_COPY_OVERLAP: return "Memory Copy Overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image Format Mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image Format Not Supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Build Program Failure";
        case CL_MAP_FAILURE: return "Map Failure";
        case CL_INVALID_VALUE: return "Invalid Value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid Device Type";
        case CL_INVALID_PLATFORM: return "Invalid Platform";
        case CL_INVALID_DEVICE: return "Invalid Device";
        case CL_INVALID_CONTEXT: return "Invalid Context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid Queue Properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid Command Queue";
        case CL_INVALID_HOST_PTR: return "Invalid Host Pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid Memory Object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid Image Format Descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid Image Size";
        case CL_INVALID_SAMPLER: return "Invalid Sampler";
        case CL_INVALID_BINARY: return "Invalid Binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid Build Options";
        case CL_INVALID_PROGRAM: return "Invalid Program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid Program Executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid Kernel Name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid Kernel Definition";
        case CL_INVALID_KERNEL: return "Invalid Kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid Argument Index";
        case CL_INVALID_ARG_VALUE: return "Invalid Argument Value";
        case CL_INVALID_ARG_SIZE: return "Invalid Argument Size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid Kernel Arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid Work Dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid Work Group Size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid Work Item Size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid Global Offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid Event Wait List";
        case CL_INVALID_EVENT: return "Invalid Event";
        case CL_INVALID_OPERATION: return "Invalid Operation";
        case CL_INVALID_GL_OBJECT: return "Invalid GL Object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid Buffer Size";
        case CL_INVALID_MIP_LEVEL: return "Invalid MIP Level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid Global Work Size";
        default: return "Unknown Error Code";
        }
    }

private:
    cl_int m_error;
};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_EXCEPTION_RUNTIME_EXCEPTION_HPP
