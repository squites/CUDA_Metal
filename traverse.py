from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_Expression, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array
from ast_builder import Parameter, Binary, Literal, CudaVar, Variable, Array

class CUDAVisitor(object): # CUDATraverse()
    """ Traverse the ast nodes """
    def visit(self, node): # 1st call node: <class '__main__.Kernel'>
        method = "visit_" + node.__class__.__name__ # attribute
        visitor = getattr(self, method, self.visit_error) # getattr(obj, attr, default): returns the value of the attribute "attr" of the object "obj". returns a reference of the function "method"
        # visitor = getattr(self, "visit_Kernel")
        # visitor() is the same as self.visit_kernel() for example
        # visitor(node): we pass the root node "Kernel()"" on 1st call.
        #print(f"visitor: {visitor} visitor method: {visit_method}")
        return visitor(node) # visit_kernel(ast_builder.Kernel)

    def visit_Kernel(self, node):
        print(f"Kernel node: {node}")
        qualifier = "kernel" if node.qualifier == "__global__" else ""
        type = node.type
        name = node.name
        
        if node.children():
            params = []
            body = []
            for child in node.children():
                print(f"Kernel child node: {child}")
                child_node = self.visit(child)
                if isinstance(child, Parameter):
                    params.append(child_node) # append(METAL_Parameter)
                else:
                    body.append(child_node) # append(METAL_Body)
            return METAL_Kernel(qualifier, type, name, params, body)
        else:
            print(f"Node {node} has no children!")
            return METAL_Kernel(qualifier, type, name, [], [])

    def visit_Parameter(self, node):
        print(f"Parameter node: {node}") # debug
        return METAL_Parameter(memory_type="device", type=node.type, name=node.name)

    def visit_Body(self, node):
        print(f"Body node: {node}") # debug
        if node.children():
            statements = []
            for child in node.children():
                print(f"Body child node: {child}")
                # for each child, will return the respective METAL node. Ex: visit(child) = METAL_Declaration(type, name, value)
                child_node = self.visit(child) # visit(Declaration), visit(Assignment)
                statements.append(child_node) # this right? It is if child_node is being a METAL node returned by visit methods. Not right if child_node is CUDA node
            #statements = [METAL_Declaration(...), METAL_Assignment(...)]
            return METAL_Body(statements)
        else:
            print(f"The node {node} has no children!")
            return METAL_Body(node.statement)

    def visit_Statement(self, node):
        pass

    def visit_Declaration(self, node): #visit_Declaration(Declaration())
        print(f"Declaration node: {node}")
        type = node.type
        name = node.name
        if node.children():
            value = [] # It'll be an Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                print(f"Declaration child node: {child}") # debug
                child_node = self.visit(child) # this is equal as: "return METAL_Declaration(type, name, value)"
                value.append(child_node)
            return METAL_Declaration(type, name, value)
        else:
            print(f"Node {node} has no children.")
            return METAL_Declaration(type, name, None)

    def visit_Assignment(self, node):
        print(f"Assignment node: {node}")
        name = self.visit(node.name) if isnode(node.name) else node.name
        val = self.visit(node.value) if isnode(node.value) else node.value
        return METAL_Assignment(name, val)

    #def visit_Expression(self, node):
    #    pass
    
    def visit_Binary(self, node):
        print(f"Binary node: {node}")
        metal_op = node.op
        left = self.visit(node.left) if isnode(node.left) else str(node.left)#self.visit(node.left) if isinstance(node.left, Binary) else self.get_expr(node.left)
        right = self.visit(node.right) if isnode(node.right) else str(node.right)#self.visit(node.right) if isinstance(node.right, Binary) else self.get_expr(node.right)
        return  METAL_Binary(metal_op, left, right)

    def visit_Literal(self, node):
        print(f"Literal node: {node}")
        type = node.type
        return METAL_Literal(type)

    def visit_Variable(self, node):
        print(f"Variable node: {node}")
        name = node.name
        return METAL_Variable(name)

    def visit_Array(self, node):
        print(f"Array node: {node}") # debug
        print(f"name: {type(node.name)}") # debug
        array_name = self.visit(node.name) if isnode(node.name) else node.name
        idx = self.visit(node.index)
        return METAL_Array(array_name, idx)

    def visit_CudaVar(self, node):
        print(f"CudaVar node: {node}")
        metal_var = metal_map(node.base)
        return METAL_Var(metal_var)

    def visit_error(self, node, attr):
        print(f"The node {node} has no attribute named {attr}!") 

# helpers (move this to another file)
def isnode(node):
    """ Check if the node that we're visiting has any node as value for any attribute """
    if isinstance(node, (Binary, Literal, Variable, Array, CudaVar)):
        return True

def metal_map(cuda_term):
    """ Maps any CUDA concept syntax into METAL concept syntax"""
    metal_term = ""
    match cuda_term:
        case "blockIdx": 
            metal_term = "[[threadgroup_position_in_grid]]"
        case "threadIdx":
            metal_term = "[[thread_position_in_threadgroup]]"
        case "blockDim":
            metal_term = "[[threads_per_threadgroup]]"
    return metal_term

def get_expr(node):
    if isinstance(node, Binary):
        metal_op = getattr(node, "op")
        metal_l = get_expr(getattr(node, "left")) 
        metal_r = get_expr(getattr(node, "right"))
        metal_node = METAL_Binary(metal_op, metal_l, metal_r)
    elif isinstance(node, Literal):
        metal_val = getattr(node, "value")
        metal_node = METAL_Literal(metal_val)
    elif isinstance(node, CudaVar):
        metal_base = getattr(node, "base")
        metal_node = METAL_Var(metal_base)
    else:
        metal_name = getattr(node, "name")
        metal_node = METAL_Variable(metal_map(metal_name))
    return metal_node

# METAL -> CUDA:
# Grid        -> Grid
# Threadgroup -> thread block
# Thread      -> thread
# SIMD_group  -> Warp
# Threadgroup memory -> SMEM
# Device memory -> GMEM
# 
# threadGroup == block
# [[thread_position_in_grid]]        == blockIdx.x * blockDim.x + threadIdx.x  # global thread index
# [[threadgroup_position_in_grid]]   == blockIdx    # block index
# [[thread_position_in_threadgroup]] == threadIdx   # thread index within the thread block
# [[threads_per_threadgroup]]        == blockDim    # dimensions of a thread block
# [[threads_per_grid]]               == blockDim * gridDim  # total thread dimension
# 
# threadgroup == __shared__
# constant    == __constant__
# gridDim   == [[threads_per_grid]]
# blockDim  == [[threads_per_threadgroup]]
# threadIdx == [[thread_position_in_threadgroup]]
#
#
# ----- METAL kernel example: -----
# kernel void add(device const float* A, 
#                 device const float* B,
#                 device float* C,
#                 uint index [[thread_position_in_grid]]) {
#   C[index] = A[index] + B[index];
# }
#
# ----- CUDA kernel example: -----
# __global__ void add (const float* A,
#                      const float* B,
#                      float *C,
#                      int arrayLen) {
#   int index = blockIdx.x * blockDim.x + threadIdx.x;
#   if (index < arrayLen) {
#       C[index] = A[index] + B[index]
#   }
# }
