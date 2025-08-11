from lark import Lark, Transformer
from grammar import cuda_grammar
from ast_builder import CUDATransformer
from traverse import CUDAVisitor
from codegen import CodeGen

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
    # TODO:
    # naive matmul kernel
    with open("./examples/naive_matmul.cu", "r") as f: # maybe create a function that generates the respective metal to all ./examples cuda kernels
        kernel_vecAdd = f.read()

    # parsing
    parser = Lark(cuda_grammar)
    parse_tree = parser.parse(kernel_vecAdd)
    #print(f"parse tree: {tree}") # type: <class 'lark.tree.Tree'>
    #print(parse_tree.pretty())
    print(parse_tree)

    # builds cuda ast
    print("CUDA ast:")
    transformer = CUDATransformer()
    cuda_ast = transformer.transform(parse_tree)
    #print(type(cuda_ast)) # type: <class '__main__.Kernel'>
    #cuda_ast.pretty_print() # structured print
    print(cuda_ast, "\n")

    # cuda visitor
    print("VISITOR:")
    cuda_visitor = CUDAVisitor()
    metal_ast = cuda_visitor.visit(cuda_ast)
    print("\nCUDA AST\n", cuda_ast)
    print("\nMETAL AST\n", metal_ast)

    # metal code gen
    #gen = CodeGen()
    #metal_code_str = gen.generator(metal_ast)
    #print(f"\nCUDA kernel:\n{kernel_vecAdd}")
    #print(f"\nMETAL kernel generated:\n{metal_code_str}")

    # writing in a file
    #filename = "vecAdd.metal"
    #with open(filename, "x") as f:
    #    f.write(metal_code_str)


if __name__ == "__main__":
    main()


# TODO:
# - generate a diverse set of cuda kernels, exploring multiple optimizations
#   using a hand-written metal dispatcher (for now)
# - then run and profile the metal kernel, comparing cuda and metal kernels results
#   and performance.