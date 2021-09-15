#include <stdio.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#define SOFTENING 1e-9f

struct Body {
    float3 xyz, vxyz;
};

__global__ void integratePositions(Body* p, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    p[i].xyz.x += p[i].vxyz.x * dt;
    p[i].xyz.y += p[i].vxyz.y * dt;
    p[i].xyz.z += p[i].vxyz.z * dt;
}

__device__ float3 bodyForce(Body* p, float dt, int i, int j) {
    float dx = p[j].xyz.x - p[i].xyz.x;
    float dy = p[j].xyz.y - p[i].xyz.y;
    float dz = p[j].xyz.z - p[i].xyz.z;
    float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;
    float3 F;
    F.x = dt * dx * invDist3; F.y = dt * dy * invDist3; F.z = dt * dz * invDist3;
    return F;
}

__device__ float3 reduce_sum_tile_shfl(thread_block_tile<32> g, float3 val) {
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val.x += g.shfl_down(val.x, i);
        val.y += g.shfl_down(val.y, i);
        val.z += g.shfl_down(val.z, i);
    }
    // note: only thread 0 will return full sum
    return val;
}

__global__ void bodyForce_tile_shfl(Body* p, float dt) {
    int i = blockIdx.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float3 my_force = bodyForce(p, dt, i, j);

    auto tile = tiled_partition<32>(this_thread_block());
    float3 tile_sum = reduce_sum_tile_shfl(tile, my_force);
    Body* sum = p + i;
    float* sumx = &(sum->vxyz.x);
    float* sumy = &(sum->vxyz.y);
    float* sumz = &(sum->vxyz.z);

    if (tile.thread_rank() == 0)
    {
        atomicAdd(sumx, tile_sum.x);
        atomicAdd(sumy, tile_sum.y);
        atomicAdd(sumz, tile_sum.z);
    }
}

__global__ void bodyForce(Body* p, float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
        float dx = p[j].xyz.x - p[i].xyz.x;
        float dy = p[j].xyz.y - p[i].xyz.y;
        float dz = p[j].xyz.z - p[i].xyz.z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vxyz.x += dt * Fx; p[i].vxyz.y += dt * Fy; p[i].vxyz.z += dt * Fz;
}

#define UNROLLED_TILED_REDUCTION

void main(const int argc, const char** argv) {

    const int nBodies = 2 << 15;
    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    float* buf;
    cudaMallocManaged(&buf, bytes);
    Body* p = (Body*)buf;

    // migrate unified memory to host before initializing (if cudaMemPrefetchAsync available for your GPU)
    //cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId);
    //:TODO: fill input array with random values, add CPU "gold" benchmark computation for verification
    //read_values_from_file(initialized_values, buf, bytes);

    // migrate unified memory to device (if cudaMemPrefetchAsync available for your GPU)
    //int deviceId;
    //cudaGetDevice(&deviceId);
    //cudaMemPrefetchAsync(p, bytes, deviceId);

    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < nIters; iter++) {

#if defined(UNROLLED_TILED_REDUCTION)
        bodyForce_tile_shfl<<<nBodies, threadsPerBlock>>>(p, dt); // outer and inner loops unrolled
#else
        bodyForce<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies); // outer loop unrolled
#endif

        integratePositions<<<numberOfBlocks, threadsPerBlock>>>(p, dt);

        cudaDeviceSynchronize();
    }

    cudaFree(buf);
}
