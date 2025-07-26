#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    constexpr int arrayLength = 20;
    NS::Error* error = nullptr;
            
    // 1) get device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr<<"Failed to create device"<<std::endl;
        return -1;
    }

    // 2) create command queue
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        std::cerr<<"Failed to create command queue"<<std::endl;
        device->release();
        return -1;
    }

    // 3) load metal library and create the function
    NS::String* filePath = NS::String::string("./vecAdd.metallib", NS::ASCIIStringEncoding);
    MTL::Library* library = device->newLibrary(filePath, &error);
    if (!library) {
        std::cerr << "Failed to create library: " << error->localizedDescription()->utf8String() << "\n";
        filePath->release();
        commandQueue->release();
        device->release();
        return -1;
    }

    
    MTL::Function* function = library->newFunction(NS::String::string("vecAdd", NS::ASCIIStringEncoding));
    if (!function) {
        std::cerr<<"Failed to create a function"<<std::endl;
        library->release();
        commandQueue->release();
        device->release();
        return -1;
    }

    // 4) create compute pipeline state
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(function, &error);
    if (!pipelineState) {
        std::cerr << "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << "\n";
        function->release();
        library->release();
        commandQueue->release();
        device->release();
        return -1;
    }

    // 5) prepare input data
    float* dataA = new float[arrayLength];
    float* dataB = new float[arrayLength];

    // fill arrays with sample data
    for (int i = 0; i < arrayLength; i++) {
        dataA[i] = static_cast<float>(i);
        dataB[i] = static_cast<float>(i*2);
    }

    // 6) create metal buffers
    MTL::Buffer* bufferA = device->newBuffer(dataA, arrayLength * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(dataB, arrayLength * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferResult = device->newBuffer(arrayLength * sizeof(float), MTL::ResourceStorageModeShared);

    // 7) create command buffer and compute encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

    // 8) set pipeline state and buffers
    computeEncoder->setComputePipelineState(pipelineState);
    computeEncoder->setBuffer(bufferA, 0, 0);
    computeEncoder->setBuffer(bufferB, 0, 1);
    computeEncoder->setBuffer(bufferResult, 0, 2);

    // 9) calculate grid and threadgroup sizes
    MTL::Size gridSize = MTL::Size(arrayLength, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(pipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);

    // 10) dispatch threads
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    // 11) end encoding, commit command buffer
    computeEncoder->endEncoding();
    commandBuffer->commit();

    // 12) wait for complete
    commandBuffer->waitUntilCompleted();

    // 13) verify results
    float* resultData = static_cast<float*>(bufferResult->contents());
    //print result
    std::cout<<"\ndataA:\n";
    for (int i = 0; i < arrayLength; i++) {
        std::cout<<dataA[i]<<"  ";
    }
    std::cout<<"\ndataB:\n";
    for (int i = 0; i < arrayLength; i++) {
        std::cout<<dataB[i]<<"  ";
    }
    std::cout<<"\nresult:\n";
    for (int i = 0; i < arrayLength; i++) {
        std::cout<<resultData[i]<<"  ";
    }

    // After verifying results, release resources
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