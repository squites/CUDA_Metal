// main.mm
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <fstream>
#include <sstream>

std::string readFile(const char* path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    // 1. Get GPU
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cout << "No GPU found" << std::endl;
        return 1;
    }
    std::cout << "GPU: " << [device.name UTF8String] << std::endl;

    // 2. Load shader
    NSError* error = nil;
    std::string shaderSource = readFile("./examples/addOne.metal");
    NSString* source = [NSString stringWithUTF8String:shaderSource.c_str()];
    
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cout << "Failed to compile: " << [[error description] UTF8String] << std::endl;
        return 1;
    }

    // 3. Get kernel function
    id<MTLFunction> function = [library newFunctionWithName:@"addOne"];

    // 4. Create pipeline
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    // 5. Create command queue
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // 6. Create data
    const int N = 1000000;//8;
    id<MTLBuffer> inbuffer = [device newBufferWithLength:N * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outbuffer = [device newBufferWithLength:N * sizeof(float) options:MTLResourceStorageModeShared];
    
    // Fill with values
    float* inputPtr = (float*)[inbuffer contents];
    for (int i = 0; i < N; i++) {
        inputPtr[i] = (float)i;
    }
    
    std::cout << "Before: ";
    for (int i = 0; i < N; i++) std::cout << inputPtr[i] << " ";
    std::cout << std::endl;

    // 7. Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // 8. Set pipeline and buffer
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:outbuffer offset:0 atIndex:0];
    [encoder setBuffer:inbuffer offset:0 atIndex:1];

    // 9. Dispatch
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    MTLSize blockSize = MTLSizeMake(N, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:blockSize];

    // 10. End and submit
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // 11. Check result
    float* outputPtr = (float*)[outbuffer contents];
    std::cout << "After:  ";
    for (int i = 0; i < N; i++) std::cout << outputPtr[i] << " ";
    std::cout << std::endl;

    return 0;
}