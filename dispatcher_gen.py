#import json

dispatcher_template = r"""
\#import <Metal/Metal.h>
\#import <Foundation/Foundation.h>
\#include <iostream>
\#include <fstream>
\#include <sstream>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newLibraryWithFile:@"kernel.metalib" error:nil];
    id<MTLFunction> func = [library newFunctionWithName:@"{kernel_name}"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction: func error: nill];
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    {buffer_bindings}

    [encoder dispatchThreads:{gridSize} threadsPerThreadgroup:{blockSize}];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    return 0;
}
"""

def gen_dispatcher(metadata):
    #with open("metadata.json", 'r') as file:
    #    metadata = json.load(file)

    # buffers
    buf_bind = ""
    for buf in metadata["buffers"]:
        buf_bind += f'[encoder setBuffer:{buf["name"]} offset:0 atIndex:{buf["idx"]}];\n'

    code = dispatcher_template.format(
        kernel_name=metadata["kernelName"],
        buffer_bindings=buf_bind,
    )

    return code



