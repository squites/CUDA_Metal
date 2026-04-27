__global__ void softmax(float* input, float* output, int n) {
    int idx = threadIdx.x;
    
    float max_val = input[0];
    for (int i = 0; i < n; i=i+1) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < n; i=i+1) {
        output[i] = expf(input[i] - max_val);
        sum = sum + output[i];
    }

    for (int i = 0; i < n; i=i+1) {
        output[i] = output[i] / sum;
    }
}