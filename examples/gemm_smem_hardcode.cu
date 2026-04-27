__global__ void gemm_smem(float* data0, float* data1, float* data2,
                                        int M, int N, int K) {
    __shared__ float data0_s[32 * 32];
    __shared__ float data1_s[32 * 32];
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;
    int tCol = threadIdx.x % 32;
    int tRow = threadIdx.x / 32;

    data0 = data0 + cRow * 32 * K;
    data1 = data1 + cCol * 32;
    data2 = data2 + cRow * 32 * N + cCol * 32;
    
    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx = bkIdx + 32) {
        data0_s[tRow * 32 + tCol] = data0[tRow * K + tCol];
        data1_s[tRow * 32 + tCol] = data1[tRow * N + tCol];
        
        __syncthreads();
        data0 = data0 + 32;
        data1 = data1 + 32 * N;
        
        for (int dotIdx = 0; dotIdx < 32; dotIdx = dotIdx + 1) {
            tmp = tmp + data0_s[tRow * 32 + dotIdx] * data1_s[dotIdx * 32 + tCol];
        }
        __syncthreads();
    }
    data2[tRow * N + tCol] = tmp;
}