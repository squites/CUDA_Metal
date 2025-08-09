__global__ void naive_matmul(float* data0, float* data1, float* data2, int M, int N, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;
    for (int i = 0; i < k; i++) {
        tmp = tmp + (data0[x * K + i] * data1[i * N + y]);
    }
    data2[x * N + y] = tmp;
}