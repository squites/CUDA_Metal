#from main import METAL_Ast
from ast_builder import METAL_Ast, METAL_Parameter

class CodeGen():
    """
    This will walk the METAL ast and generate a string which is the metal code 
    """
    #def __init__(self, metal_ast):
    #    self.metal_ast = metal_ast

    def generator(self, node):
        method = "gen_" + str(node)
        gen = getattr(self, method)
        return gen(node)

    def gen_METAL_Parameter(self, node):
        memory_type = node.memory_type
        type = node.type
        name = node.name
        code_str = f"{str(memory_type)} {str(type)} {str(name)};"
        return code_str 
        
#ast = CodeGen()
#node = METAL_Parameter(memory_type="device", type="int*", name="a")
#code = ast.gen_METAL_Parameter(node)
#print(f"CODE GENERATED: {code}")
    
"""
# CUDA vecAdd:
__global__ void vecAdd(int* a,
                       int* b,
                       int* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

# METAL vecAdd:
kernel void vecAdd(device const int* a [[buffer(0)]],
                   device const int* b [[buffer(1)]],
                   device int* c [[buffer(2)]],
                   uint idx [[thread_position_in_grid]]) {
    c[idx] = a[idx] + b[idx];
}

"""