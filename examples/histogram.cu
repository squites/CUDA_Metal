__global__ void histogram(int* data, int* hist, int size, int buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int res = data[idx];
        atomicAdd(&hist[res], 1);
    }
}