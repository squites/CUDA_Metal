__global__ void addOne(float* out, float* in) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in[i] + 1;
}