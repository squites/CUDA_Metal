from lark import Lark, Transformer
from grammar import cuda_grammar
from ast_builder import CUDATransformer
from traverse import CUDAVisitor
from codegen import CodeGen
from dispatcher_gen import gen_dispatcher

import subprocess
import numpy as np

import argparse
import json
import os

#def validate_input(path: str):
#def alloc_device_mem():
#def move_to_device():
#def initialize_data():
#def devicefree():
#def hostfree():

def main():
    # add flags to control to execution and pass Grid and Block size. Also flags for to print the CUDA AST, Metal AST ...
    arg = argparse.ArgumentParser()
    arg.add_argument("cuda_path", type=str)
    arg.add_argument("--grid", type=str, default="1,1,1")
    arg.add_argument("--block", type=str, default="1,1,1")
    arg.add_argument("-N", type=int, default=1)
    arg.add_argument("--dims", type=int, default=1)
    args = arg.parse_args()

    with open(args.cuda_path, "r") as f:
        cudakernel = f.read()

    grid = [int(x) for x in args.grid.split(",")]
    block = [int(x) for x in args.block.split(",")]
    N = args.N
    kernel_name = os.path.splitext(os.path.basename(args.cuda_path))[0]

    # parsing
    parser = Lark(cuda_grammar)
    parse_tree = parser.parse(cudakernel)
    #print(parse_tree.pretty()) # type: <class 'lark.tree.Tree'>

    # builds cuda ast
    transformer = CUDATransformer() # type: <class 'ast_builder.CUDA_Program'>
    cuda_ast = transformer.transform(parse_tree)
    print(f"{cuda_ast}\n")

    # generates metal ast
    cuda_visitor = CUDAVisitor()
    metal_ast = cuda_visitor.visit(cuda_ast)

    totalSize = N ** args.dims # if dims==2, then its a matrix
    cuda_visitor.kernel_metadata["launch_config"] = {
        "grid": grid,
        "block": block,
        "N": args.N,
        "dims": args.dims,
        "totalSize": totalSize#N ** len(cuda_visitor.thread_idx_dims)
    }
    cuda_visitor.kernel_metadata["kernelFile"] = kernel_name
    with open("metadata.json", 'w') as json_file:
        json.dump(cuda_visitor.kernel_metadata, json_file, indent=2)

    # generate dispatcher code
    dispatcher_code = gen_dispatcher(cuda_visitor.kernel_metadata)
    with open("dispatcher.mm", "w") as f:
        f.write(dispatcher_code)

    #print("\nCUDA AST\n", cuda_ast)
    print("\nMETAL AST\n", metal_ast)

    # generates metal code
    gen = CodeGen(cuda_visitor.thread_idx_dims) # pass here the mappings: CodeGen(cuda_visitor.thread_idx_dims)
    metal_kernel = gen.generator(metal_ast)
    print(f"\nCUDA kernel:\n", cudakernel)
    print(f"\nMETAL Shader generated:\n{metal_kernel}")

    # writing in a file
    filename = f"./examples/{kernel_name}.metal" #"./examples/addOne.metal"
    with open(filename, "w") as f:
        f.write(metal_kernel)

    # 2. compile metal shader
    subprocess.run([
        "xcrun", "-sdk", "macosx", "metal", "-c", f"./examples/{kernel_name}.metal", "-o", f"{kernel_name}.air"
    ])
    subprocess.run([
        "xcrun", "-sdk", "macosx", "metallib", f"{kernel_name}.air", "-o", f"{kernel_name}.metallib"
    ])

    # 3. compile dispatcher
    subprocess.run([
        "clang++", "-framework", "Metal", "-framework", "Foundation", "dispatcher.mm", "-o", "runner"
    ])

    # 4. generate input data (needs to be duplicated if we have 2 inputs. For example, for data0 and data1, each size 15, we need to generate then 30 values)
    data = []
    for p in cuda_visitor.kernel_metadata["kernel"]["buffers"]:
        if p["access"] == "read":
            #data.append(np.ones(totalSize).astype(np.int32))
            data.append(np.random.rand(totalSize).astype(np.float32))
    
    print(f'input {data}\ndata shape: {len(data)} buffers\n{totalSize} elements each')
    #print(f'first buffer 10 values {data[0][:10]}')
    np.concatenate(data).tofile("input.bin")

    # 5. run kernel
    subprocess.run(["./runner"])
    #subprocess.run(["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--launch", "./runner"])

    # 6. read output (find a way to automatically set the np type instead of handcoded!)
    #output = np.fromfile("output.bin", dtype=np.float32)#.reshape(64,64)
    output = np.fromfile("output.bin", dtype=np.float32)
    print("Output:", output)

if __name__ == "__main__":
    main()

# TODO:
# - profile metal kernel, comparing with cuda results
#
# - OBS: kernel "gemm_smem.cu" is not working. For some reason its only computing the first 2 columns of the matrix
# Tried changind the grid to 1D/2D, didnt work. Tried changing the kernel dims to use threadIdx.y as well, and 
# the result remained the same. Not sure why. Maybe there's an emulation to be implemented here.
# (GPT): Apple GPUs execute threads in SIMD groups (typically 32 threads). Large 1-D threadgroups behave 
# differently than CUDA.
#Common Metal threadgroup shapes:
#(32,32,1)
#(16,16,1)
#(8,8,8)
#
#Using (1024,1,1)
# is technically allowed but not idiomatic and sometimes problematic because the scheduler organizes 
# execution in simdgroups.
#
#So CUDA flattening patterns don't always translate cleanly!  
# 
# 
# 
# 
# OBS:
# For now, we can keep kernels unsafe, and need to enforce correctness in the dispatcher
# Meaning we need to validate inputs before launching
# ensure:
# - dtype matches
# - values are in range
# - buffers sized correctly
#
# TODO: 
# - add `size`, that shows how many elements are stored in the buffer. Can be for example: `size: "N"`, doesn't
# have to be a explicit number. can be a variable.
# 
# - For knowing if the values are inside the boundaries, gpt told me to do a minimal pattern detection. So find
# patterns like `hist[something]`, trace the source like `hist[data[i]]` or `int res = data[i]; hist[res]`.
# attach constraint, so when we detect "buffer A is used as index into buffer B", generate a constraint on A,
# telling "values in A [0, sizeof(B)]" This is Constraint Inference Pass. So implement a value tracking across 
# nodes. 
# All of this is basically for ensuring that any value used as memory index stays within the bounds of the
# allocated buffer. This is important because on GPU when out-of-bounds happen, there's a silent corruption, so
# you dont really know what happend. 
#
# IDEA: we only check values that originate from buffers. Examples:
# a) hist[data[i]]: direct buffer used as index
# b)    int res = data[i];
#       hist[res];
#   This is a scalar `res` but originated from buffer `data`, so the constrained is applied to `data`
#
# When there's no need to track:
# c) (X) don't check this
#       int res = 5
#       hist[res] 
#   Here there's no buffer involved, so we don't care.
#
# So how to do this:
# 1) during traversal look for indexing patterns like:
#    int res = data[i]
#    hist[res]
#   
# OR
#   hist[data[i]]
# Where, buffer `data` is used as index to buffer `hist`
# 
# 2) trace the buffer, so trace the buffer `data` from variable `res`
# 3) emit metadata: constraint: {src=data, target=hist, type=index_relatioN}
#
# During dispatcher:
# 1) for each constraint, get src and target buffers.
# 2) compute size of target
# 3) validate: assert that value < buckets