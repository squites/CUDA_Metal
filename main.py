from lark import Lark, Transformer
from grammar import cuda_grammar
from ast_builder import CUDATransformer, METAL_Ast #CUDA_ast
from traverse import CUDAVisitor#, metal_mapping
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
    # move this to a .cu file 
    kernel_vecAdd = r"""
    __global__ void vecAdd(int* a, int* b, int* c) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        c[idx] = a[idx] + b[idx];  
    }
    """
    # parsing
    parser = Lark(cuda_grammar)
    parse_tree = parser.parse(kernel_vecAdd)
    #print(f"parse tree: {tree}") # type: <class 'lark.tree.Tree'>
    #print(tree.pretty())

    # builds cuda ast
    print("CUDA ast:")
    transformer = CUDATransformer()
    cuda_ast = transformer.transform(parse_tree)
    #print(type(cuda_ast)) # type: <class '__main__.Kernel'>
    cuda_ast.pretty_print() # structured print
    print(cuda_ast)
    print("\n")

    # cuda visitor
    print("VISITOR:")
    cuda_visitor = CUDAVisitor()
    metal_ast = cuda_visitor.visit(cuda_ast)
    print("\nMETAL AST\n", metal_ast)
    print("\nCUDA AST\n", cuda_ast)

    # metal code gen
    print("\nMETAL code generated\n")
    #gen = CodeGen()
    #code_str = gen.generator(metal_ast)
    #print(code_str)

if __name__ == "__main__":
    main()