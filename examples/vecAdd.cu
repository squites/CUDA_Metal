__global__ void vecAdd(float* data0, float* data1, float* data2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data2[idx] = data0[idx] + data1[idx];
    }
}