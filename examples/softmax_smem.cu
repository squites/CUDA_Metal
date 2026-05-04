__global__ void softmax_smem(float* input, float* output, int n) {
    __shared__ float smem[256];
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    float local_max = input[0];
    for (int i = tid; i < n; i+=blockSize) {
        if (input[i] > local_max) {
            local_max = input[i];
        }
    }
    smem[tid] = local_max;
    __syncthreads();
    
    for (int s = blockSize / 2; s > 0; s = s/2) {
        if (tid < s) {
            if (smem[tid + s] > smem[tid]) {
                smem[tid] = smem[tid + s];
            }
        }
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();
    
    float local_sum = 0.0;
    for (int i = tid; i < n; i += blockSize) {
        float val = expf(input[i] - max_val);
        output[i] = val;
        local_sum += val;
    }
    smem[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockSize / 2; s > 0; s=s/2) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float sum = smem[0];
    __syncthreads();
    
    for (int i = tid; i < n; i += blockSize) {
        output[i] = output[i] / sum;
    }
}