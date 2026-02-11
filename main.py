from lark import Lark, Transformer
from grammar import cuda_grammar
from ast_builder import CUDATransformer
from traverse import CUDAVisitor
from codegen import CodeGen

import argparse

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
    # TODO: add flags to control to execution. Flags for which example to run, to print the tree, ... and so on.
    #arg = argparse.ArgumentParser()
    #arg.add_argument("-t", "--tree", action=)

    with open("./examples/addOne.cu", "r") as f:
    #with open("./examples/gemm_smem_cacheblocking.cu", "r") as f: # maybe create a function that generates the respective metal to all ./examples cuda kernels
        cudakernel = f.read()

    # parsing
    parser = Lark(cuda_grammar)
    parse_tree = parser.parse(cudakernel)
    #print(f"parse tree: {tree}") # type: <class 'lark.tree.Tree'>
    #print(parse_tree.pretty())
    #print(parse_tree)

    # builds cuda ast
    transformer = CUDATransformer()
    cuda_ast = transformer.transform(parse_tree)
    #print(type(cuda_ast)) # type: <class 'ast_builder.CUDA_Program'>
    print(f"{cuda_ast}\n")

    # cuda visitor
    #print("VISITOR:")
    cuda_visitor = CUDAVisitor()
    metal_ast = cuda_visitor.visit(cuda_ast)
    #print("PARAMS: ")#cuda_visitor.kernel_params)
    #for p in cuda_visitor.kernel_params:
    #    print(" ",p)
    #print("\nCUDA AST\n", cuda_ast)
    print("\nMETAL AST\n", metal_ast)

    # metal code gen
    gen = CodeGen()
    metal_kernel = gen.generator(metal_ast)
    #print(f"\nCUDA kernel:\n{kernel_vecAdd}")
    print(f"\nMETAL Shader generated:\n{metal_kernel}")

    # writing in a file
    filename = "./examples/addOne.metal"
    with open(filename, "x") as f:
        f.write(metal_kernel)

if __name__ == "__main__":
    main()


# TODO:
# - run and profile the metal kernel, comparing cuda and metal kernels results
#   and performance.