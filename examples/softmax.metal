#include <metal_stdlib>
using namespace metal;
kernel void softmax(device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant int& n [[buffer(2)]],
                         uint3 tid_local [[thread_position_in_threadgroup]]) {
    int idx = tid_local.x;
    float max_val = input[0];
    for (int i = 0; i < n; i = i + 1) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    float sum = 0.0;
    for (int i = 0; i < n; i = i + 1) {
        output[i] = exp(input[i] - max_val);
        sum = sum + output[i];
    }
    for (int i = 0; i < n; i = i + 1) {
        output[i] = output[i] / sum;
    }
}