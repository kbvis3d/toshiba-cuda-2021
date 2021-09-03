#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
#define DIVUP(a, b) ((a)+(b)-1)/(b)
#define ITEMS_PER_THREAD 16
//#define SIMPLE
//#define MULTIPLE_ITEMS_PER_THREAD
#define MULTIPLE_ITEMS_PER_THREAD_STRIDED

__global__ void
vectorAddSimple(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
__global__ void
vectorAddMultipleItems(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int k = 0; k < ITEMS_PER_THREAD; ++k)
	{
		if (i < numElements)
		{
			C[i] = A[i] + B[i];
		}
	}
}
__global__ void
vectorAddMultipleItemsStrided(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x * ITEMS_PER_THREAD + threadIdx.x;

    for (int k = 0;k < ITEMS_PER_THREAD; ++k)
    {
        if (i < numElements)
        {
            C[i] = A[i] + B[i];
            i += blockDim.x;
        }
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Print the vector length to be used, and compute its size
    int numElements = 2 << 24;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the input and output vectors
    float* A, * B, * C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i)
    {
        A[i] = rand()/(float)RAND_MAX;
        B[i] = rand()/(float)RAND_MAX;
    }

    // Launch the Vector Add CUDA Kernel

    // query device properties
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int multiProcessorCount = prop.multiProcessorCount;
    int maxBlocksPerMultiProcessor = prop.maxBlocksPerMultiProcessor;
    int maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    printf("multiProcessorCount:%d, maxBlocksPerMultiProcessor:%d, maxThreadsPerMultiProcessor:%d\n", multiProcessorCount, maxBlocksPerMultiProcessor, maxThreadsPerMultiProcessor);

#if defined(SIMPLE)
    int threadsPerBlock = 32; // not enough threads for occupancy
    //int threadsPerBlock = 256;
    //int threadsPerBlock = 1024;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddSimple<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
#elif defined(MULTIPLE_ITEMS_PER_THREAD)
    int threadsPerBlock = 256;
    int blocksPerGrid = (DIVUP(numElements, ITEMS_PER_THREAD) + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddMultipleItems<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
#elif defined(MULTIPLE_ITEMS_PER_THREAD_STRIDED)
    int threadsPerBlock = 256;
    int blocksPerGrid = (DIVUP(numElements, ITEMS_PER_THREAD) + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddMultipleItemsStrided<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
#endif

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    printf("Done\n");
    return 0;
}

