dispatcher_template = r"""#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <fstream>
#include <sstream>

int main() {{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newLibraryWithFile:@"{lib_file}" error:nil];
    id<MTLFunction> function = [library newFunctionWithName:@"{kernel_name}"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:nil];
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    const int dataSize = {data_size};
    {buffer_creation}
    
    std::ifstream input("input.bin", std::ios::binary);
    {buffer_fill}
    input.close();

    [encoder setComputePipelineState:pipeline];

    {buffer_bindings}
    {scalar_bindings}
    MTLSize gridSize = MTLSizeMake({grid_config});
    MTLSize blockSize = MTLSizeMake({block_config});
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    std::ofstream output("output.bin", std::ios::binary);
    {buffer_out}
    output.close();

    return 0;
}}
"""
# create function to fill buffers with values to execute the kernel (the size can be passed together
# with grid and block sizes)
def gen_dispatcher(metadata):
    space = " " * 4
    
    # buffer creation
    buf_create = ""
    for buf in metadata["kernel"]["buffers"]:
        buf_create += f'id<MTLBuffer> {buf["name"]}Buffer = [device newBufferWithLength:dataSize * sizeof(float) options:MTLResourceStorageModeShared];\n{space}'
        
    # fill buffers 
    buf_fill = ""
    for buf in metadata["kernel"]["buffers"]:
        if buf["access"] == "read":
            buf_fill += f'input.read((char*)[{buf["name"]}Buffer contents], dataSize * sizeof(float));\n'
    
    # buffer bindings
    buf_bind = ""
    for buf in metadata["kernel"]["buffers"]:
        buf_bind += f'[encoder setBuffer:{buf["name"]}Buffer offset:0 atIndex:{buf["idx"]}];\n{space}'

    # scalar bindings
    scalar_bind = ""
    for scalar in metadata["kernel"]["scalars"]:
        scalar_bind += f'{scalar["type"]} {scalar["name"]} = dataSize;\n{space}'
        scalar_bind += f'[encoder setBytes:&{scalar["name"]} length:sizeof({scalar["type"]}) atIndex:{scalar["idx"]}];\n{space}'

    # grid and block
    g = metadata["launch_config"]["grid"]
    b = metadata["launch_config"]["block"]
    grid = f'{g[0]}, {g[1]}, {g[2]}'
    block = f'{b[0]}, {b[1]}, {b[2]}'

    buf_out = ""
    for buf in metadata["kernel"]["buffers"]:
        if buf["access"] == "write":
            buf_out += f'output.write((char*)[{buf["name"]}Buffer contents], dataSize*sizeof(float));\n'


    code = dispatcher_template.format(
        kernel_name=metadata["kernel"]["kernelName"],
        lib_file=f'{metadata["kernel"]["kernelName"]}.metallib',
        buffer_creation=buf_create,
        buffer_fill=buf_fill,
        buffer_bindings=buf_bind,
        scalar_bindings=scalar_bind,
        grid_config=grid,
        block_config=block,
        data_size=metadata["launch_config"]["dataSize"],
        buffer_out=buf_out,
    )

    return code



