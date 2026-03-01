dispatcher_template = r"""#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <fstream>
#include <sstream>

int main() {{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newLibraryWithFile:@"{lib_file}" error:nil];
    id<MTLFunction> function = [library newFunctionWithName:@"{kernel_name}"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    {buffer_bindings}
    MTLSize gridSize = MTLSizeMake({grid_config});
    MTLSize blockSize = MTLSizeMake({block_config});
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    return 0;
}}
"""
# create function to fill buffers with values to execute the kernel


def gen_dispatcher(metadata):
    # buffers
    buf_bind = ""
    space = " " * 4
    for buf in metadata["kernel"]["buffers"]:
        buf_bind += f'[encoder setBuffer:{buf["name"]} offset:0 atIndex:{buf["idx"]}];\n{space}'

    # grid and block
    g = metadata["launch_config"]["grid"]
    b = metadata["launch_config"]["block"]
    grid = f'{g[0]}, {g[1]}, {g[2]}'
    block = f'{b[0]}, {b[1]}, {b[2]}'

    code = dispatcher_template.format(
        kernel_name=metadata["kernel"]["kernelName"],
        lib_file=f'{metadata["kernel"]["kernelName"]}.metallib',
        buffer_bindings=buf_bind,
        grid_config=grid,
        block_config=block,
    )

    return code



