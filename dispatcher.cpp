#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

float randFloat() {
    return (float)(std::rand()) / (float)(std::rand());    
}

int main() {
    constexpr int arrayLength = 20;
    NS::Error* error = nullptr;
            
    // 1) get device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    // 2) create command queue
    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // 3) load metal library
    NS::String* filePath = NS::String::string("./naive_matmul.metallib", NS::ASCIIStringEncoding);
    MTL::Library* library = device->newLibrary(filePath, &error);

    // 4) create function 
    MTL::Function* function = library->newFunction(NS::String::string("naive_matmul", NS::ASCIIStringEncoding));

    // 5) create compute pipeline state
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(function, &error);

    // 6) prepare input data
    float* dataA = new float[arrayLength];
    float* dataB = new float[arrayLength];

    // fill arrays with data (todo: create func to generate random float numbers)
    srand(time(0));
    for (int i = 0; i < arrayLength; i++) {
        dataA[i] = randFloat();
        dataB[i] = randFloat();
        //dataA[i] = static_cast<float>(i);
        //dataB[i] = static_cast<float>(i*2);
    }

    // 7) create metal buffers
    MTL::Buffer* bufferA = device->newBuffer(dataA, arrayLength * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(dataB, arrayLength * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferResult = device->newBuffer(arrayLength * sizeof(float), MTL::ResourceStorageModeShared);

    // 8) create command buffer and compute encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

    // 9) set pipeline state and buffers
    computeEncoder->setComputePipelineState(pipelineState);
    computeEncoder->setBuffer(bufferA, 0, 0); // (buffer, offset: 0, buffer: 0), find a way to tell this index for the kernel
    computeEncoder->setBuffer(bufferB, 0, 1);
    computeEncoder->setBuffer(bufferResult, 0, 2);

    // 10) calculate grid and threadgroup sizes
    MTL::Size gridSize = MTL::Size(arrayLength, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(pipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);

    // 11) dispatch threads
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    // 12) end encoding, commit command buffer
    computeEncoder->endEncoding();
    commandBuffer->commit();

    // 13) wait for complete
    commandBuffer->waitUntilCompleted();

    // verify results
    float* resultData = static_cast<float*>(bufferResult->contents());
    std::cout<<"\ndataA:\n";
    for (int i = 0; i < arrayLength; i++) {
        std::cout<<dataA[i]<<" ";
    }
    std::cout<<"\ndataB:\n";
    for (int i = 0; i < arrayLength; i++) {
        std::cout<<dataB[i]<<" ";
    }
    std::cout<<"\nresult:\n";
    for (int i = 0; i < arrayLength; i++) {
        std::cout<<resultData[i]<<" ";
    }

    // release resources (free)
    bufferA->release();
    bufferB->release();
    bufferResult->release();
    pipelineState->release();
    function->release();
    library->release();
    commandQueue->release();
    device->release();
    delete[] dataA;
    delete[] dataB;

    return 0;
}

/*
TODO:
- implement function to check if the result is right instead of printing the array
*/