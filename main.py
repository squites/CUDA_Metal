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
    #arg.add_argument("--dataSize", type=str, default="1024")
    #arg.add_argument("-S1", type=int, default=1)
    #arg.add_argument("-S2", type=int, default=1)
    #arg.add_argument("-S3", type=int, default=1)
    args = arg.parse_args()

    #with open("./examples/addOne.cu", "r") as f:
    # maybe create a function that generates the respective metal to all ./examples cuda kernels
    with open(args.cuda_path, "r") as f:
        cudakernel = f.read()

    grid = [int(x) for x in args.grid.split(",")]
    block = [int(x) for x in args.block.split(",")]
    #dataDim = [int(x) for x in args.dataSize.split(",")]
    N = args.N
    #for d in dataDim:
    #    totalSize *= d
    #totalSize = args.S1 * args.S2 * args.S3
    kernel_name = os.path.splitext(os.path.basename(args.cuda_path))[0]

    # parsing
    parser = Lark(cuda_grammar)
    parse_tree = parser.parse(cudakernel)
    #print(parse_tree.pretty()) # type: <class 'lark.tree.Tree'>

    # builds cuda ast
    transformer = CUDATransformer() # type: <class 'ast_builder.CUDA_Program'>
    cuda_ast = transformer.transform(parse_tree)
    #print(f"{cuda_ast}\n")

    # generates metal ast
    cuda_visitor = CUDAVisitor()
    metal_ast = cuda_visitor.visit(cuda_ast)
    
    # Better: 1-Generate dispatcher once, 2-Pass grid/block at runtime. 3-Remove from metadata
    # ideally, the grid and block size wouldn't be on json, and should be only passed during runtime
    #cuda_visitor.kernel_metadata["launch_config"] = {"grid": grid, "block": block, "dataSize": dataDim, "totalSize": totalSize}
    totalSize = N ** len(cuda_visitor.thread_idx_dims)
    cuda_visitor.kernel_metadata["launch_config"] = {
        "grid": grid,
        "block": block,
        "N": args.N,
        #"scalar1": args.S1,
        #"scalar2": args.S2,
        #"scalar3": args.S3,
        "totalSize": totalSize#N ** len(cuda_visitor.thread_idx_dims)
    }
    cuda_visitor.kernel_metadata["kernelFile"] = kernel_name
    #print(cuda_visitor.kernel_metadata)
    with open("metadata.json", 'w') as json_file:
        json.dump(cuda_visitor.kernel_metadata, json_file, indent=2)

    print("THREAD DIMS USED :")
    for k,v in cuda_visitor.thread_idx_dims.items():
        print(f"{k}: {v}")
    print("N:", N)
    print("totalSize:", totalSize)

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
            #data.append(np.arange(args.dataSize, dtype=np.float32))
            data.append(np.random.rand(totalSize).astype(np.float32))
    
    print(f'input {data}\n')
    np.concatenate(data).tofile("input.bin")

    # 5. run kernel
    subprocess.run(["./runner"])
    #subprocess.run(["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--launch", "./runner"])

    # 6. read output
    output = np.fromfile("output.bin", dtype=np.float32)
    print("Output:", output)

if __name__ == "__main__":
    main()


# TODO:
# - run and profile the metal kernel, comparing cuda and metal kernels results
#   and performance.
#
# - add a nice print after running the kernel, showing how many threads, blocks, etc. Or figure it out the profiler