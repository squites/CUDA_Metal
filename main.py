from lark import Lark, Transformer
from grammar import cuda_grammar
from ast_builder import CUDATransformer
from traverse import CUDAVisitor
from codegen import CodeGen
from dispatcher_gen import gen_dispatcher

import argparse
import json

#def validate_input(path: str):
#    pass
#
#def alloc_device_mem():
#    pass
#
#def move_to_device():
#    pass
#
#def initialize_data():
#    pass
#
#def devicefree():
#    pass
#
#def hostfree():
#    pass

def main():
    # add flags to control to execution and pass Grid and Block size. Also flags for to print the CUDA AST, Metal AST ...
    arg = argparse.ArgumentParser()
    arg.add_argument("--grid", type=int, nargs=3, default=[1,1,1])
    arg.add_argument("--block", type=int, nargs=3, default=[1,1,1])
    args = arg.parse_args()

    # flag to pass the filepath?
    with open("./examples/addOne.cu", "r") as f:
    # maybe create a function that generates the respective metal to all ./examples cuda kernels
        cudakernel = f.read()

    # parsing
    parser = Lark(cuda_grammar)
    parse_tree = parser.parse(cudakernel)
    #print(parse_tree.pretty()) # type: <class 'lark.tree.Tree'>

    # builds cuda ast
    transformer = CUDATransformer()
    cuda_ast = transformer.transform(parse_tree)
    #print(type(cuda_ast)) # type: <class 'ast_builder.CUDA_Program'>
    print(f"{cuda_ast}\n")

    # visit cuda ast
    cuda_visitor = CUDAVisitor()
    metal_ast = cuda_visitor.visit(cuda_ast)
    
    # Better: 1-Generate dispatcher once, 2-Pass grid/block at runtime. 3-Remove from metadata
    # ideally, the grid and block size wouldn't be on json, and should be only passed during runtime
    cuda_visitor.kernel_metadata["launch_config"] = {"grid": args.grid, "block": args.block}
    with open("metadata.json", 'w') as json_file:
        json.dump(cuda_visitor.kernel_metadata, json_file, indent=2)

    # generate dispatcher code
    dispatcher_code = gen_dispatcher(cuda_visitor.kernel_metadata)
    with open("dispatcher.mm", "w") as f:
        f.write(dispatcher_code)

    #print("\nCUDA AST\n", cuda_ast)
    print("\nMETAL AST\n", metal_ast)

    # metal code gen
    gen = CodeGen()
    metal_kernel = gen.generator(metal_ast)
    print(f"\nMETAL Shader generated:\n{metal_kernel}")

    # writing in a file
    filename = "./examples/addOne.metal"
    with open(filename, "w") as f:
        f.write(metal_kernel)

if __name__ == "__main__":
    main()


# TODO:
# - run and profile the metal kernel, comparing cuda and metal kernels results
#   and performance.
#   
# - cool feature would be ask the user to type gridSize and blockSize in the command line while compiling
# the code. If the user doesn't input anything, we generate the metal code, but test with many different
# grid and block sizes, and select the best one (faster). 