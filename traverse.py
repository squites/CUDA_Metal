from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_IfStatement, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array
from ast_builder import Parameter, Binary, Literal, CudaVar, Variable, Array

class CUDAVisitor(object): # CUDATraverse()
    """ Traverse the ast nodes """
    def __init__(self):
        #self.buffer_idx = -1
        self.kernel_params = []
        self.body = []
    
    def visit(self, node, idx=0): # 1st call node: <class '__main__.Kernel'>
        method = "visit_" + node.__class__.__name__ # attribute
        visitor = getattr(self, method, self.visit_error) # getattr(obj, attr, default): returns the value of the attribute "attr" of the object "obj". returns a reference of the function "method"
        # visitor = getattr(self, "visit_Kernel")
        # visitor() is the same as self.visit_kernel() for example
        # visitor(node): we pass the root node "Kernel()"" on 1st call.
        if str(method) == "visit_Parameter": # find a way to remove this and simplify this recursive function
            return visitor(node, idx) # index for [[buffer(idx)]]
        print(f"method: {method}, visitor: {visitor}")
        return visitor(node) # same as: return visit_kernel(ast_builder.Kernel)

    def visit_Kernel(self, node):
        #print(f"Kernel node: {node}")
        qualifier = "kernel" if node.qualifier == "__global__" else ""
        type = node.type
        name = node.name
        
        if node.children():
            #params = []
            param_idx = -1
            body = []
            for child in node.children():
                #print(f"Kernel child node: {child}")
                param_idx += 1
                child_node = self.visit(child, idx=param_idx)
                if isinstance(child, Parameter):
                    self.kernel_params.append(child_node)
                else:
                    body.append(child_node) 
            return METAL_Kernel(qualifier, type, name, self.kernel_params, body)
        else:
            #print(f"Node {node} has no children!")
            return METAL_Kernel(qualifier, type, name, [], [])

    def visit_Parameter(self, node, buffer_idx=0):
        #print(f"Parameter node: {node}") # debug
        mem_type = metal_map(node.mem_type)
        if node.type == "int*" or node.type == "float*":
            #buffer_idx = self.kernel_params.index(node) #get_param_idx(self.kernel_params, node) #+= 1
            buffer = f"[[buffer({buffer_idx})]]"
        else:
            buffer = None # when its not a vector, matrix or tensor
        return METAL_Parameter(memory_type=mem_type, type=node.type, name=node.name, buffer=buffer)

    def visit_Body(self, node):
        #print(f"Body node: {node}") # debug
        if node.children():
            statements = []
            for child in node.children():
                #print(f"Body child node: {child}")
                # for each child, will return the respective METAL node. Ex: visit(child) = METAL_Declaration(type, name, value)
                child_node = self.visit(child) # visit(Declaration), visit(Assignment)
                statements.append(child_node) # this right? It is if child_node is being a METAL node returned by visit methods. Not right if child_node is CUDA node
            #statements = [METAL_Declaration(...), METAL_Assignment(...)]
            return METAL_Body(statements)
        else:
            #print(f"The node {node} has no children!")
            return METAL_Body(node.statement)

    def visit_Statement(self, node):
        pass

    def visit_Declaration(self, node): #visit_Declaration(Declaration())
        #print(f"Declaration node: {node}")
        type = node.type
        name = node.name
        if node.children():
            value = [] # = Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                #print(f"Declaration child node: {child}") # debug
                child_node = self.visit(child) # this is equal as: "return METAL_Declaration(type, name, value)"
                value.append(child_node)
            return METAL_Declaration(type, name, value)
        else:
            #print(f"Node {node} has no children.")
            return METAL_Declaration(type, name, None)

    def visit_Assignment(self, node):
        #print(f"Assignment node: {node}")
        name = self.visit(node.name) if isnode(node.name) else node.name
        val = self.visit(node.value) if isnode(node.value) else node.value
        return METAL_Assignment(name, val)

    def visit_IfStatement(self, node):
        print(f"If node: {node}")
        condition = node.condition
        body = []
        for child in node.children():
            child_node = self.visit(child)
            body.append(child_node)
        return METAL_IfStatement(condition=condition, if_body=body)

    #def visit_Expression(self, node):
    #    pass
    
    def visit_Binary(self, node):
        #print(f"Binary node: {node}")
        metal_op = node.op
        left = self.visit(node.left) if isnode(node.left) else str(node.left)
        right = self.visit(node.right) if isnode(node.right) else str(node.right)
        return  METAL_Binary(metal_op, left, right)

    def visit_Literal(self, node):
        #print(f"Literal node: {node}")
        type = node.type
        return METAL_Literal(type)

    def visit_Variable(self, node):
        #print(f"Variable node: {node}")
        name = node.name
        return METAL_Variable(name)

    def visit_Array(self, node):
        #print(f"Array node: {node}") # debug
        #print(f"name: {type(node.name)}") # debug
        array_name = self.visit(node.name) if isnode(node.name) else node.name
        idx = self.visit(node.index)
        return METAL_Array(array_name, idx)

    def visit_CudaVar(self, node):
        #print(f"CudaVar node: {node}")
        metal_var = metal_map(node.base)
        return METAL_Var(metal_var)

    def visit_error(self, node, attr):
        print(f"The node {node} has no attribute named {attr}!") 

# helpers (move this to another file)
def isnode(node):
    """ Check if the node that we're visiting has any node as value for any attribute """
    if isinstance(node, (Binary, Literal, Variable, Array, CudaVar,
                         METAL_Binary, METAL_Literal, METAL_Variable, 
                         METAL_Array, METAL_Var)):
        return True

def metal_map(cuda_term):
    """ Maps any CUDA concept syntax into METAL concept syntax"""
    metal_term = ""
    match cuda_term:
        case "blockIdx":     metal_term = "[[threadgroup_position_in_grid]]"
        case "threadIdx":    metal_term = "[[thread_position_in_threadgroup]]"
        case "blockDim":     metal_term = "[[threads_per_threadgroup]]"
        case "__global__":   metal_term = "device"
        case "__shared__":   metal_term = "threadgroup"
        case "__constant__": metal_term = "constant"
    
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

def get_param_idx(kernel_params, node):
    for p in kernel_params:
        if node.type == p.type and node.name == p.name:
            print(f"index: {int(p)}")
            return int(p)
    return 0

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
