__global__ void naive_matmul(float* data0, float* data1, float* data2, int M, int N, int K) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tx < M) && (ty < N)) {
        float tmp = 0.0;
        for (int i = 0; i < K; i=i+1) {
            tmp = tmp + (data0[tx * K + i] * data1[i * N + ty]);
        }
        data2[tx * N + ty] = tmp;
    }
}