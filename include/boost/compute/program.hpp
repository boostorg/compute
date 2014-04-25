//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_PROGRAM_HPP
#define BOOST_COMPUTE_PROGRAM_HPP

#include <string>
#include <vector>
#include <fstream>
#include <streambuf>

#ifdef BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <iostream>
#endif

#include <boost/move/move.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/detail/assert_cl_success.hpp>
#include <boost/compute/detail/program_create_kernel_result.hpp>

#ifdef BOOST_COMPUTE_USE_OFFLINE_CACHE
#include <sstream>
#include <boost/optional.hpp>
#include <boost/filesystem.hpp>
#include <boost/compute/platform.hpp>
#include <boost/compute/detail/getenv.hpp>
#include <boost/compute/detail/sha1.hpp>
#endif

namespace boost {
namespace compute {

/// \class program
/// \brief A compute program.
///
/// The program class represents an OpenCL program.
///
/// \see kernel
class program
{
public:
    /// Creates a null program object.
    program()
        : m_program(0)
    {
    }

    /// Creates a program object for \p program. If \p retain is \c true,
    /// the reference count for \p program will be incremented.
    explicit program(cl_program program, bool retain = true)
        : m_program(program)
    {
        if(m_program && retain){
            clRetainProgram(m_program);
        }
    }

    /// Creates a new program object as a copy of \p other.
    program(const program &other)
        : m_program(other.m_program)
    {
        if(m_program){
            clRetainProgram(m_program);
        }
    }

    /// \internal_
    program(BOOST_RV_REF(program) other)
        : m_program(other.m_program)
    {
        other.m_program = 0;
    }

    /// Copies the program object from \p other to \c *this.
    program& operator=(const program &other)
    {
        if(this != &other){
            if(m_program){
                clReleaseProgram(m_program);
            }

            m_program = other.m_program;

            if(m_program){
                clRetainProgram(m_program);
            }
        }

        return *this;
    }

    /// \internal_
    program& operator=(BOOST_RV_REF(program) other)
    {
        if(this != &other){
            if(m_program){
                clReleaseProgram(m_program);
            }

            m_program = other.m_program;
            other.m_program = 0;
        }

        return *this;
    }

    /// Destroys the program object.
    ~program()
    {
        if(m_program){
            BOOST_COMPUTE_ASSERT_CL_SUCCESS(
                clReleaseProgram(m_program)
            );
        }
    }

    /// Returns the underlying OpenCL program.
    cl_program& get() const
    {
        return const_cast<cl_program &>(m_program);
    }

    /// Returns the source code for the program.
    std::string source() const
    {
        return get_info<std::string>(CL_PROGRAM_SOURCE);
    }

    /// Returns the binary for the program.
    std::vector<unsigned char> binary() const
    {
        size_t binary_size = get_info<size_t>(CL_PROGRAM_BINARY_SIZES);
        std::vector<unsigned char> binary(binary_size);

        unsigned char *binary_ptr = &binary[0];
        cl_int error = clGetProgramInfo(m_program,
                                        CL_PROGRAM_BINARIES,
                                        sizeof(unsigned char **),
                                        &binary_ptr,
                                        0);
        if(error != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }

        return binary;
    }

    std::vector<device> get_devices() const
    {
        size_t device_ids_size = 0;
        clGetProgramInfo(m_program,
                         CL_PROGRAM_DEVICES,
                         0,
                         0,
                         &device_ids_size);

        std::vector<cl_device_id> device_ids(device_ids_size / sizeof(size_t));
        clGetProgramInfo(m_program,
                         CL_PROGRAM_DEVICES,
                         device_ids_size,
                         &device_ids[0],
                         0);

        std::vector<device> devices;
        for(size_t i = 0; i < device_ids.size(); i++){
            devices.push_back(device(device_ids[i]));
        }

        return devices;
    }

    /// Returns the context for the program.
    context get_context() const
    {
        return context(get_info<cl_context>(CL_PROGRAM_CONTEXT));
    }

    /// Returns information about the program.
    ///
    /// \see_opencl_ref{clGetProgramInfo}
    template<class T>
    T get_info(cl_program_info info) const
    {
        return detail::get_object_info<T>(clGetProgramInfo, m_program, info);
    }

    /// Builds the program with \p options.
    void build(const std::string &options = std::string())
    {
        const char *options_string = 0;

        if(!options.empty()){
            options_string = options.c_str();
        }

        cl_int ret = clBuildProgram(m_program, 0, 0, options_string, 0, 0);

        #ifdef BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
        if(ret != CL_SUCCESS){
            // print the error, source code and build log
            std::cerr << "Boost.Compute: "
                      << "kernel compilation failed (" << ret << ")\n"
                      << "--- source ---\n"
                      << source()
                      << "\n--- build log ---\n"
                      << build_log()
                      << std::endl;
        }
        #endif

        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }
    }

    /// Returns the build log.
    std::string build_log() const
    {
        device device = get_devices()[0];

        size_t size = 0;
        cl_int ret = clGetProgramBuildInfo(m_program,
                                           device.id(),
                                           CL_PROGRAM_BUILD_LOG,
                                           0,
                                           0,
                                           &size);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        std::string value(size - 1, 0);
        ret = clGetProgramBuildInfo(m_program,
                                    device.id(),
                                    CL_PROGRAM_BUILD_LOG,
                                    size,
                                    &value[0],
                                    0);
        if(ret != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(ret));
        }

        return value;
    }

    /// Creates and returns a new kernel object for \p name.
#ifndef BOOST_COMPUTE_DOXYGEN_INVOKED
    detail::program_create_kernel_result
#else
    kernel
#endif
    create_kernel(const std::string &name) const
    {
        cl_int error = 0;
        cl_kernel kernel = clCreateKernel(m_program,
                                          name.c_str(),
                                          &error);
        if(!kernel){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }

        return detail::program_create_kernel_result(kernel);
    }

    /// \internal_
    operator cl_program() const
    {
        return m_program;
    }

    /// Creates a new program with \p source in \p context.
    ///
    /// \see_opencl_ref{clCreateProgramWithSource}
    static program create_with_source(const std::string &source,
                                      const context &context)
    {
        const char *source_string = source.c_str();

        cl_int error = 0;
        cl_program program_ = clCreateProgramWithSource(context,
                                                        uint_(1),
                                                        &source_string,
                                                        0,
                                                        &error);
        if(!program_){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }

        return program(program_, false);
    }

    /// Creates a new program with \p file in \p context.
    ///
    /// \see_opencl_ref{clCreateProgramWithSource}
    static program create_with_source_file(const std::string &file,
                                           const context &context)
    {
        // open file stream
        std::ifstream stream(file.c_str());

        // read source
        std::string source(
            (std::istreambuf_iterator<char>(stream)),
            std::istreambuf_iterator<char>()
        );

        // create program
        return create_with_source(source, context);
    }

    /// Creates a new program with \p binary of \p binary_size in
    /// \p context.
    ///
    /// \see_opencl_ref{clCreateProgramWithBinary}
    static program create_with_binary(const unsigned char *binary,
                                      size_t binary_size,
                                      const context &context)
    {
        const cl_device_id device = context.get_device().id();

        cl_int error = 0;
        cl_int binary_status = 0;
        cl_program program_ = clCreateProgramWithBinary(context,
                                                        uint_(1),
                                                        &device,
                                                        &binary_size,
                                                        &binary,
                                                        &binary_status,
                                                        &error);
        if(!program_){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }
        if(binary_status != CL_SUCCESS){
            BOOST_THROW_EXCEPTION(runtime_exception(binary_status));
        }

        return program(program_, false);
    }

    /// Creates a new program with \p binary in \p context.
    ///
    /// \see_opencl_ref{clCreateProgramWithBinary}
    static program create_with_binary(const std::vector<unsigned char> &binary,
                                      const context &context)
    {
        return create_with_binary(&binary[0], binary.size(), context);
    }

    /// Creates a new program with \p file in \p context.
    ///
    /// \see_opencl_ref{clCreateProgramWithBinary}
    static program create_with_binary_file(const std::string &file,
                                           const context &context)
    {
        // open file stream
        std::ifstream stream(file.c_str(), std::ios::in | std::ios::binary);

        // read binary
        std::vector<unsigned char> binary(
            (std::istreambuf_iterator<char>(stream)),
            std::istreambuf_iterator<char>()
        );

        // create program
        return create_with_binary(&binary[0], binary.size(), context);
    }

    /// Create a new program with \p source in \p context and builds it with \p options.
    /**
     * In case BOOST_COMPUTE_USE_OFFLINE_CACHE macro is defined,
     * the compiled binary is stored for reuse in the offline cache located in
     * $HOME/.boost_compute on UNIX-like systems and in %APPDATA%/boost_compute
     * on Windows.
     */
    static program build_with_source(
            const std::string &source,
            const context     &context,
            const std::string &options = std::string()
            )
    {
#ifdef BOOST_COMPUTE_USE_OFFLINE_CACHE
        // Get hash string for the kernel.
        std::string hash;
        {
            device   d(context.get_device());
            platform p(d.get_info<cl_platform_id>(CL_DEVICE_PLATFORM));

            std::ostringstream src;
            src << "// " << p.name() << " v" << p.version() << "\n"
                << "// " << context.get_device().name() << "\n"
                << "// " << options << "\n\n"
                << source;

            hash = detail::sha1(src.str());
        }

        // Try to get cached program binaries:
        try {
            boost::optional<program> prog = load_program_binary(hash, context);

            if (prog) {
                prog->build(options);
                return *prog;
            }
        } catch (...) {
            // Something bad happened. Fallback to normal compilation.
        }

        // Cache is apparently not available. Just compile the sources.
#endif
        const char *source_string = source.c_str();

        cl_int error = 0;
        cl_program program_ = clCreateProgramWithSource(context,
                                                        uint_(1),
                                                        &source_string,
                                                        0,
                                                        &error);
        if(!program_){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }

        program prog(program_, false);
        prog.build(options);

#ifdef BOOST_COMPUTE_USE_OFFLINE_CACHE
        // Save program binaries for future reuse.
        save_program_binary(hash, prog);
#endif

        return prog;
    }

private:
    BOOST_COPYABLE_AND_MOVABLE(program)

    cl_program m_program;

#ifdef BOOST_COMPUTE_USE_OFFLINE_CACHE
    // Path delimiter symbol for the current OS.
    static const std::string& path_delim() {
        static const std::string delim =
            boost::filesystem::path("/").make_preferred().string();
        return delim;
    }

    // Path to appdata folder.
    static inline const std::string& appdata_path() {
#ifdef WIN32
        static const std::string appdata = detail::getenv("APPDATA")
            + path_delim() + "boost_compute";
#else
        static const std::string appdata = detail::getenv("HOME")
            + path_delim() + ".boost_compute";
#endif
        return appdata;
    }

    // Path to cached binaries.
    static std::string program_binary_path(const std::string &hash, bool create = false)
    {
        std::string dir = appdata_path()    + path_delim()
                        + hash.substr(0, 2) + path_delim()
                        + hash.substr(2);

        if (create) boost::filesystem::create_directories(dir);

        return dir + path_delim();
    }

    // Saves program binaries for future reuse.
    static void save_program_binary(const std::string &hash, const program &prog)
    {
        std::string fname = program_binary_path(hash, true) + "kernel";
        std::ofstream bfile(fname.c_str(), std::ios::binary);
        if (!bfile) return;

        std::vector<unsigned char> binary = prog.binary();

        size_t binary_size = binary.size();
        bfile.write((char*)&binary_size, sizeof(size_t));
        bfile.write((char*)binary.data(), binary_size);
    }

    // Tries to read program binaries from file cache.
    static boost::optional<program> load_program_binary(
            const std::string &hash, const context &ctx
            )
    {
        std::string fname = program_binary_path(hash) + "kernel";
        std::ifstream bfile(fname.c_str(), std::ios::binary);
        if (!bfile) return boost::optional<program>();

        size_t binary_size;
        std::vector<unsigned char> binary;

        bfile.read((char*)&binary_size, sizeof(size_t));

        binary.resize(binary_size);
        bfile.read((char*)binary.data(), binary_size);

        return boost::optional<program>(
                program::create_with_binary(
                    binary.data(), binary_size, ctx
                    )
                );
    }
#endif // BOOST_COMPUTE_USE_OFFLINE_CACHE

};

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_PROGRAM_HPP
