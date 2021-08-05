#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define BLOCKSIZE 256
#define RADIUS 3

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void stencil(const int N, int* in, int* out)
{
    __shared__ int temp[BLOCKSIZE + 2 * RADIUS];
    int gIndex = threadIdx.x + blockIdx.x * BLOCKSIZE;
    int lIndex = threadIdx.x + RADIUS;

    if (gIndex < N)
    {
        // read input data into shared memory
        temp[lIndex] = in[gIndex + RADIUS];
        if (threadIdx.x < RADIUS)
        {
            temp[lIndex - RADIUS] = in[gIndex];
            temp[lIndex + BLOCKSIZE] = in[gIndex + RADIUS + BLOCKSIZE];
        }
    }

    // ensure all reads are complete
    __syncthreads();

    if (gIndex < N - 2 * RADIUS)
    {
        // apply the stencil
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; ++offset)
        {
            result += temp[lIndex + offset];
        }

        // output the filtered result
        out[gIndex + RADIUS] = result;
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Print the vector length to be used, and compute its size
    const int numElements = 1000000;
    size_t size = numElements * sizeof(int);
    printf("[Filtering of %d elements with filter radius of %d]\n", numElements, RADIUS);

    // Allocate the input and output vectors
    int* in, * out;
    cudaMallocManaged(&in, size);
    cudaMallocManaged(&out, size);

    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i)
    {
        in[i] = i % 10;
    }

    // launch stencil kernel
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((numElements - 2 * RADIUS) / blockSize.x + 1);
    stencil<<<gridSize, blockSize>>> (numElements, in, out);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Verify that the filtered output is correct
    for (int i = RADIUS; i < numElements - RADIUS; ++i)
    {
        int result = 0;
        for (int j = -RADIUS;j <= RADIUS;++j)
        {
            result += in[i + j];
        }
        if (fabs(result - out[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    cudaFree(in);
    cudaFree(out);

    return 0;
}

