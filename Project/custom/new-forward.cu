#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int TILE_WIDTH = 16;

    // Shared memory tiles
    __shared__ float tile_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_mask[K][K];
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

     // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    int row_i = row_o - K / 2;
    int col_i = col_o - K / 2;

    // Initialize output value for each thread
    float output_val = 0.0f;

    // Loop over tiles
    for (int m = 0; m < M; m++) {
        for (int c = 0; c < C; c++) {
            // Load input tile into shared memory
            if ((row_i >= 0) && (row_i < H) && (col_i >= 0) && (col_i < W))
                tile_input[ty][tx] = in_4d(0, c, row_i, col_i);
            else
                tile_input[ty][tx] = 0.0f;

            // Load mask tile into shared memory
            if (ty < K && tx < K)
                tile_mask[ty][tx] = mask_4d(m, c, ty, tx);

            __syncthreads();

            // Perform convolution on the tiles
            if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
                for (int i = 0; i < K; ++i) {
                    for (int j = 0; j < K; ++j) {
                        output_val += tile_mask[i][j] * tile_input[ty + i][tx + j];
                    }
                }
            }

            __syncthreads();
        }
        if (row_o < H_out && col_o < W_out)
            out_4d(0, m, row_o, col_o) += output_val;
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    // Allocate memory for device_input, device_output, device_mask
    cudaMalloc(device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc(device_output_ptr, B * M * ((H - K) / S + 1) * ((W - K) / S + 1) * sizeof(float));
    cudaMalloc(device_mask_ptr, M * C * K * K * sizeof(float));
    
    // Copy data from host_input, host_output, host_mask to device
    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize device_output to zero
    cudaMemset(*device_output_ptr, 0, B * M * ((H - K) / S + 1) * ((W - K) / S + 1) * sizeof(float));
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
       // Calculate grid and block sizes
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, (H_out + blockDim.y - 1) / blockDim.y, (M + blockDim.z - 1) / blockDim.z);
    
    // Launch the conv_forward_kernel
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    
    // Check for errors after kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy output from device to host_output
    cudaMemcpy(host_output, device_output, B * M * ((H - K) / S + 1) * ((W - K) / S + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory for device_input, device_output, device_mask
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
    
    // Check for errors after copy and free
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
