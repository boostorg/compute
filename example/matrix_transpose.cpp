#include <iostream>
#include <cstdlib>

#include <boost/compute.hpp>
#include <boost/compute/type_traits/type_name.hpp>

namespace compute = boost::compute;

#define TILE_DIM 32
#define BLOCK_ROWS 8

/// \fn _copyKernel
/// \brief generate a copy kernel program
compute::program _copyKernel(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void copy_kernel(__global const float *src, __global float *dst)
        {
            uint x = get_group_id(0) * TILE_DIM + get_local_id(0);
            uint y = get_group_id(1) * TILE_DIM + get_local_id(1);
            
            uint width = get_num_groups(0) * TILE_DIM;
            
            for(uint i = 0 ; i < TILE_DIM ; i+= BLOCK_ROWS)
            {
                dst[(y+i)*width +x] = src[(y+i)*width + x];
            }
        }
    );
    // create copy program
    std::stringstream options;
    options << "-DTILE_DIM=" << TILE_DIM
            << " -DBLOCK_ROWS=" << BLOCK_ROWS;
    compute::program program = compute::program::build_with_source(source,context,options.str());
    return program;
}

/// \fn _naiveTransposeKernel
/// \brief generate a naive transpose kernel program
compute::program _naiveTransposeKernel(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void naiveTranspose(__global const float *src, __global float *dst)
        {
            uint x = get_group_id(0) * TILE_DIM + get_local_id(0);
            uint y = get_group_id(1) * TILE_DIM + get_local_id(1);
            
            uint width = get_num_groups(0) * TILE_DIM;
            
            for(uint i = 0 ; i < TILE_DIM; i+= BLOCK_ROWS)
            {
                dst[x*width + y+i] = src[(y+i)*width + x];
            }
        }  
   );
    
   // create naiveTranspose program
    std::stringstream options;
    options << "-DTILE_DIM=" << TILE_DIM
            << " -DBLOCK_ROWS=" << BLOCK_ROWS;
    compute::program program = compute::program::build_with_source(source,context,options.str());
    return program;
}

/// \fn _coalescedTransposeKernel
/// \brief generate a coalesced transpose kernel program
compute::program _coalescedTransposeKernel(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void coalescedTranspose(__global const float *src, __global float *dst)
        {
            __local float tile[TILE_DIM][TILE_DIM];
            
            // compute indexes
            uint x = get_group_id(0) * TILE_DIM + get_local_id(0);
            uint y = get_group_id(1) * TILE_DIM + get_local_id(1);
            
            uint width = get_num_groups(0) * TILE_DIM;
            
            // load inside local memory
            for(uint i = 0 ; i < TILE_DIM; i+= BLOCK_ROWS)
            {
                tile[get_local_id(1)+i][get_local_id(0)] = src[(y+i)*width + x];
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // transpose indexes
            x = get_group_id(1) * TILE_DIM + get_local_id(0);
            y = get_group_id(0) * TILE_DIM + get_local_id(1);
            
            // write output from local memory
            for(uint i = 0 ; i < TILE_DIM ; i+=BLOCK_ROWS)
            {
                dst[(y+i)*width + x] = tile[get_local_id(0)][get_local_id(1)+i];
            }
            
        }  
   );
    
   // create naiveTranspose program
    std::stringstream options;
    options << "-DTILE_DIM=" << TILE_DIM
            << " -DBLOCK_ROWS=" << BLOCK_ROWS;
    compute::program program = compute::program::build_with_source(source,context,options.str());
    return program;
}

/// \fn _coalescedNoBankConflictsKernel
/// \brief generate a coalesced withtout bank conflicts kernel program
compute::program _coalescedNoBankConflictsKernel(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void coalescedNoBankConflicts(__global const float *src, __global float *dst)
        {
            // TILE_DIM+1 is here to avoid bank conflicts in local memory
            __local float tile[TILE_DIM][TILE_DIM+1];
            
            // compute indexes
            uint x = get_group_id(0) * TILE_DIM + get_local_id(0);
            uint y = get_group_id(1) * TILE_DIM + get_local_id(1);
            
            uint width = get_num_groups(0) * TILE_DIM;
            
            // load inside local memory
            for(uint i = 0 ; i < TILE_DIM; i+= BLOCK_ROWS)
            {
                tile[get_local_id(1)+i][get_local_id(0)] = src[(y+i)*width + x];
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // transpose indexes
            x = get_group_id(1) * TILE_DIM + get_local_id(0);
            y = get_group_id(0) * TILE_DIM + get_local_id(1);
            // write output from local memory
            for(uint i = 0 ; i < TILE_DIM ; i+=BLOCK_ROWS)
            {
                dst[(y+i)*width + x] = tile[get_local_id(0)][get_local_id(1)+i];
            }
            
        }  
   );
    
   // create naiveTranspose program
    std::stringstream options;
    options << "-DTILE_DIM=" << TILE_DIM
            << " -DBLOCK_ROWS=" << BLOCK_ROWS;
    compute::program program = compute::program::build_with_source(source,context,options.str());
    return program;
}

/// \fn _checkTransposition
/// \brief Compare @a expectedResult to @a transposedMatrix
bool _checkTransposition(const std::vector<float>& expectedResult, 
                         uint size, 
                         const std::vector<float>& transposedMatrix)
{
    for(uint i = 0 ; i < size ; ++i)
    {
        if(expectedResult[i] != transposedMatrix[i])
        {
            std::cout << "idx = " << i << " , expected " << expectedResult[i] <<
            " , got " << transposedMatrix[i] << std::endl;
            std::cout << "FAILED" << std::endl;
            return false;
        }
    }
    return true;
}

/// \fn _generateMatrix
/// \brief generate a matrix inside @a in and do the tranposition inside @a transposeRef
void _generateMatrix(std::vector<float>& in, std::vector<float>& transposeRef, uint nx, uint ny)
{
    // generate a matrix
    for(uint i = 0 ; i < nx ; ++i)
    {
        for(uint j = 0 ; j < ny ; ++j)
        {
            in[i*ny + j] = i*ny + j; 
        }
    }
    
    // store transposed result
    for(uint j = 0; j < ny ; ++j)
    {
        for(uint i = 0 ; i < nx ; ++i)
        {
            transposeRef[j*nx + i] = in[i*ny + j];
        }
    }
}

#define _BEGIN_TEST(name) std::cout << name << std::endl;
#define _END    std::cout << std::endl;


int main(int argc, char *argv[])
{
    // Check the command line
    if(argc < 3){
        std::cerr << "Dimensions: mat[x * 1024][y * 1024] \n Usage: " << argv[0]
                  << " x y " << std::endl;
        return -1;
    }

    const uint nx = 1024 * atoi(argv[1]);
    const uint ny = 1024 * atoi(argv[2]);
    
    std::cout << "Float Matrix Size : " << nx << "x" << ny<<std::endl;
    std::cout << "Size in MB : "<< (nx * ny * sizeof(float))/1024/1024 <<std::endl;
    std::cout << "Grid Size  : " << nx/TILE_DIM << "x" << ny/TILE_DIM << " blocks" << std::endl;
    std::cout << "Local Size : " << TILE_DIM << "x" << BLOCK_ROWS << " threads" << std::endl;
    
    std::cout << std::endl;
    
    const size_t g_workSize[2] = {nx,ny*BLOCK_ROWS/TILE_DIM};
    const size_t l_workSize[2] = {TILE_DIM,BLOCK_ROWS};
    
    const uint size = nx * ny;
    try {
	    std::vector<float> h_input(size);
	    std::vector<float> h_output(size);
	    std::vector<float> expectedResult(size);

	    _generateMatrix(h_input,expectedResult,nx,ny);
	    
	    // get the default device
	    compute::device device = compute::system::default_device();
	    
	    // create a context for the device
	    compute::context context(device);

	    // device vectors
	    compute::vector<float> d_input(size,context);
	    compute::vector<float> d_output(size,context);
	    
	    // command_queue with profiling
	    compute::command_queue queue(context, device,compute::command_queue::enable_profiling);
	    
	    // copy input data
	    compute::copy(h_input.begin(),h_input.end(),d_input.begin(),queue);
	    
	    compute::program copy_program = _copyKernel(context);
	    compute::kernel kernel(copy_program,"copy_kernel");
	    kernel.set_arg(0,d_input);
	    kernel.set_arg(1,d_output);
	    
	    compute::event start;
	    _BEGIN_TEST("Copy_Kernel");
	    start = queue.enqueue_nd_range_kernel(kernel,2,0,g_workSize,l_workSize);
	    queue.finish();
	    uint64_t elapsed = start.duration<boost::chrono::nanoseconds>().count();
	    
	    std::cout << "\tElapsed: " << elapsed  << " ns"<< std::endl;
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  / ( elapsed / 1000)  << " MB/s" << std::endl; 
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  /  elapsed   << " GB/s" << std::endl; 
	    compute::copy(d_output.begin(),d_output.end(),h_output.begin(),queue);
	    
	    if(_checkTransposition(h_input,nx*ny,h_output))
		std::cout << "\tStatus: Success" << std::endl;
	    else
		std::cout << "\tStatus: Error" << std::endl;
	    _END
	    
	    _BEGIN_TEST("naiveTranspose")
	    
	    kernel = compute::kernel(_naiveTransposeKernel(context),"naiveTranspose");
	    kernel.set_arg(0,d_input);
	    kernel.set_arg(1,d_output);
	    
	    start = queue.enqueue_nd_range_kernel(kernel,2,0,g_workSize,l_workSize);
	    queue.finish();
	    elapsed = start.duration<boost::chrono::nanoseconds>().count();
	    std::cout << "\tElapsed: " << elapsed  << " ns"<< std::endl;
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  / ( elapsed / 1000)  << " MB/s" << std::endl; 
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  /  elapsed   << " GB/s" << std::endl; 
	    compute::copy(d_output.begin(),d_output.end(),h_output.begin(),queue);
	    
	    if(_checkTransposition(expectedResult,nx*ny,h_output))
		std::cout << "\tStatus: Success" << std::endl;
	    else
		std::cout << "\tStatus: Error" << std::endl;
	    _END
	    
	    _BEGIN_TEST("coalescedTranspose")
	    
	    kernel = compute::kernel(_coalescedTransposeKernel(context),"coalescedTranspose");
	    kernel.set_arg(0,d_input);
	    kernel.set_arg(1,d_output);
	    
	    start = queue.enqueue_nd_range_kernel(kernel,2,0,g_workSize,l_workSize);
	    queue.finish();
	    elapsed = start.duration<boost::chrono::nanoseconds>().count();
	    std::cout << "\tElapsed: " << elapsed  << " ns"<< std::endl;
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  / ( elapsed / 1000)  << " MB/s" << std::endl; 
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  /  elapsed   << " GB/s" << std::endl;     
	    
	    compute::copy(d_output.begin(),d_output.end(),h_output.begin(),queue);
	    
	    if(_checkTransposition(expectedResult,nx*ny,h_output))
		std::cout << "\tStatus: Success" << std::endl;
	    else
		std::cout << "\tStatus: Error" << std::endl;
	    _END
	    
	    _BEGIN_TEST("coalescedNoBankConflicts")
	    
	    kernel = compute::kernel(_coalescedNoBankConflictsKernel(context),"coalescedNoBankConflicts");
	    kernel.set_arg(0,d_input);
	    kernel.set_arg(1,d_output);
	    
	    start = queue.enqueue_nd_range_kernel(kernel,2,0,g_workSize,l_workSize);
	    queue.finish();
	    elapsed = start.duration<boost::chrono::nanoseconds>().count();
	    std::cout << "\tElapsed: " << elapsed  << " ns"<< std::endl;
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  / ( elapsed / 1000)  << " MB/s" << std::endl; 
	    std::cout << "\tBandWidth: " << 2 * nx * ny *sizeof(float)  /  elapsed   << " GB/s" << std::endl;     
	    
	    compute::copy(d_output.begin(),d_output.end(),h_output.begin(),queue);
	    
	    if(_checkTransposition(expectedResult,nx*ny,h_output))
		std::cout << "\tStatus: Success" << std::endl;
	    else
		std::cout << "\tStatus: Error" << std::endl;
	    _END
    }
    catch(boost::compute::opencl_error &e)
    {
         std::cout << "What went wrong: " << e.error_string() << std::endl;
    }
    return 0;
}


