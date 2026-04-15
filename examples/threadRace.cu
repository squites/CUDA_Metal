__global__ void threadRace(int* winner) {
    int idx = threadIdx.x;
    int old = atomicCAS(winner, -1, idx);
}