#include <metal_stdlib>
using namespace metal;

kernel void vecAdd(device float* a [[buffer(0)]],
                   device float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   uint idx [[thread_position_in_grid]]) {
    c[idx] = a[idx] + b[idx];
}