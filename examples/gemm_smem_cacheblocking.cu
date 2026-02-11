__global__ void gemm_smem_cacheblocking(float* data0, float* data1, float* data2,
                                        int M, int N, int K, int CHUNKSIZE) {
    __shared__ float data0_s[CHUNKSIZE * CHUNKSIZE];
    __shared__ float data1_s[CHUNKSIZE * CHUNKSIZE];
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;
    int tCol = threadIdx.x % CHUNKSIZE;
    int tRow = threadIdx.x / CHUNKSIZE;

    data0 = data0 + cRow * CHUNKSIZE * K;
    data1 = data1 + cCol * CHUNKSIZE;
    data2 = data2 + cRow * CHUNKSIZE * N + cCol * CHUNKSIZE;
    
    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx = bkIdx + CHUNKSIZE) {
        data0_s[tRow * CHUNKSIZE + tCol] = data0[tRow * K + tCol];
        data1_s[tRow * CHUNKSIZE + tCol] = data1[tRow * N + tCol];
        __syncthreads();
        data0 = data0 + CHUNKSIZE;
        data1 = data1 + CHUNKSIZE * N;
        for (int dotIdx = 0; dotIdx < CHUNKSIZE; dotIdx = dotIdx + 1) {
            tmp = tmp + data0_s[tRow * CHUNKSIZE + dotIdx] * data1_s[dotIdx * CHUNKSIZE + tCol];
        }
    }
    data2[tRow * N + tCol] = tmp;
}