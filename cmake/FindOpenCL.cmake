# Find OpenCL
#
# This will define the following variables:
#
#   OpenCL_FOUND        - TRUE if OpenCL was found on the system
#   OpenCL_LIBRARY      - Path to the OpenCL library (e.g. libOpenCL.so)
#   OpenCL_INCLUDE_DIR  - Include path for OpenCL

# find OpenCL library
find_library(
  OpenCL_LIBRARY
  OpenCL
  DOC "OpenCL library"
)

# find OpenCL include directory
if(APPLE)
  find_path(
    OpenCL_INCLUDE_DIR
    OpenCL/cl.h
    DOC "OpenCL include path"
  )
else()
  find_path(
    OpenCL_INCLUDE_DIR
    CL/cl.h
    DOC "OpenCL include path"
    /usr/include/nvidia-current
    /usr/include/nvidia-experimental-310/
    /opt/nvidia/cuda/include
  )
endif()

# check for the cl_ext.h header
if(EXISTS "${OpenCL_INCLUDE_DIR}/CL/cl_ext.h")
  set(OpenCL_HEADER_CL_EXT_FOUND)
endif()

# check for the cl_gl.h header
if(EXISTS "${OpenCL_INCLUDE_DIR}/CL/cl_gl.h")
  set(OpenCL_HEADER_CL_GL_FOUND)
endif()

# set OpenCL_INCLUDE_DIRS and OpenCL_LIBRARIES
set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})

# handle find_package() args
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenCL
  DEFAULT_MSG
  OpenCL_LIBRARY OpenCL_INCLUDE_DIR
)
mark_as_advanced(OpenCL_LIBRARY OpenCL_INCLUDE_DIR)

# set OpenCL_FOUND variable
set(OpenCL_FOUND ${OPENCL_FOUND})
