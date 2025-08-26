from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_IfStatement, METAL_ForStatement, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Program
from ast_builder import Parameter, Declaration, Binary, Literal, CudaVar, Variable, Array

class CUDAVisitor(object):
    """ Traverse the ast nodes """
    def __init__(self):
        #self.buffer_idx = -1
        self.kernel_params = []
        self.body = []

    # passing the node parent
    def visit(self, node, parent=None, idx=0):
        #print(f"PARENT: {str(parent)}")
        #print(f"NODE: {str(node)}")
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.visit_error)
        if str(method) == "visit_Parameter":
            return visitor(node, idx)
        if parent is not None:
            return visitor(node, parent)
        else:
            return visitor(node)

    def visit_CUDA_Program(self, node):
        lib = node.header
        kernel = self.visit(node.kernel)
        return METAL_Program(header=lib, kernel=kernel)

    def visit_Kernel(self, node):
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
        if node.children():
            statements = []
            for child in node.children():
                # for each child, will return the respective METAL node. Ex: visit(child) = METAL_Declaration(type, name, value)
                child_node = self.visit(child) # visit(Declaration), visit(Assignment)
                # check if its Parameter() node for tid and gid
                #if isinstance(child_node, METAL_Parameter):
                #    print("DECLARATIONNNN", child_node)
                
                statements.append(child_node) # this right? It is if child_node is being a METAL node returned by visit methods. Not right if child_node is CUDA node
            #statements = [METAL_Declaration(...), METAL_Assignment(...)]
            print("STATEMENTS: ", statements)
            return METAL_Body(statements)
        else:
            #print(f"The node {node} has no children!")
            return METAL_Body(node.statement)

    def visit_Statement(self, node):
        pass

    # obs: for tid and gid, we are generating the right Parameter() node, but they are still inside the 
    # body of the kernel. We need to have a way to move them into Parameters when they're thread index 
    # calculations. 
    def visit_Declaration(self, node, parent=None): #visit_Declaration(Declaration())
        cudaVarsList = []
        type = node.type
        name = node.name
        if node.children():
            value = [] # = Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                check_semantic(node.value, cudaVarsList)
                if cudaVarsList:
                    param = metal_map(" ".join(cudaVarsList))
                    node = METAL_Parameter(memory_type=None, type="uint", name=name, buffer=None, init=param)
                    self.kernel_params.append(node)
                    return None # this returns None. I need to return nothing!
                    #return METAL_Parameter(memory_type=None, type="uint", name=name, buffer=None, init=param)
                child_node = self.visit(child, parent=node) # this is equal as: "return METAL_Declaration(type, name, value)"
                value.append(child_node)
            return METAL_Declaration(type, name, value)
        else:
            return METAL_Declaration(type, name, None)

    def visit_Assignment(self, node, parent=None):
        name = self.visit(node.name, parent=node) if isnode(node.name) else node.name
        val = self.visit(node.value, parent=node) if isnode(node.value) else node.value
        return METAL_Assignment(name, val)

    def visit_IfStatement(self, node, parent=None):
        cond = self.visit(node.condition, parent=node)
        body = []
        if node.children():
            for child in node.children():
                child_node = self.visit(child, parent=node)
                body.append(child_node)
        return METAL_IfStatement(condition=cond, if_body=body)

    def visit_ForStatement(self, node, parent=None):
        # initialization, condition, increment, body
        init = self.visit(node.init, parent=node)
        cond = self.visit(node.condition, parent=node)
        incr = self.visit(node.increment, parent=node)
        stmts = []
        for child in node.children():
            child_node = self.visit(child, parent=node)
            stmts.append(child_node)
        return METAL_ForStatement(init=init, condition=cond, increment=incr, forBody=stmts, parent=parent)

    def visit_Binary(self, node, parent=None):
        self.parent = node if parent is not None else None
        metal_op = node.op
        left = self.visit(node.left, parent=node) if isnode(node.left) else str(node.left)
        right = self.visit(node.right, parent=node) if isnode(node.right) else str(node.right)
        return  METAL_Binary(metal_op, left, right)

    def visit_Literal(self, node, parent=None):
        value = node.value
        return METAL_Literal(value=value)

    def visit_Variable(self, node, parent=None):
        name = node.name
        return METAL_Variable(name)

    def visit_Array(self, node, parent=None):
        array_name = self.visit(node.name, parent=node) if isnode(node.name) else node.name
        idx = self.visit(node.index)
        return METAL_Array(array_name, idx)

    def visit_CudaVar(self, node, parent=None):
        metal_var = metal_map(node.base)
        return METAL_Var(metal_var)

    def visit_error(self, node, attr):
        print(f"The node {node} has no attribute named {attr}!") 

# helpers (move this to another file)
def isnode(node):
    """ Check if the node that we're visiting has any node as value for any attribute """
    return isinstance(node, (Binary, Literal, Variable, Array, CudaVar, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Var))

def metal_map(cuda_term):
    """ Maps CUDA concept syntax into METAL concept syntax"""
    metal_term = ""
    match cuda_term:
        case "blockIdx":     metal_term = "[[threadgroup_position_in_grid]]"
        case "threadIdx":    metal_term = "[[thread_position_in_threadgroup]]"
        case "blockDim":     metal_term = "[[threads_per_threadgroup]]"
        case "__global__":   metal_term = "device"
        case "__shared__":   metal_term = "threadgroup"
        case "__constant__": metal_term = "constant"
        case "blockIdx.x * blockDim.x + threadIdx.x": metal_term = "[[thread_position_in_grid]]"
        case "blockIdx.y * blockDim.y + threadIdx.y": metal_term = "[[thread_position_in_threadgroup]]"
    
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
            return int(p)
    return 0

# node.value = Bin(op='+', left=Bin(op='*', left=CudaVar(base='blockIdx', dim='x'), right=CudaVar(base='blockDim', dim='x')), right=CudaVar(base='threadIdx', dim='x'))
#def check_semantic2(node, cudaVars, op=None): # param: node.value
#    # not right yet. I need to return an expression instead of a list, and see if that expression 
#    # matches the correct calculation of a global thread index for example. So I need to find a way 
#    # to check if that expression is actually calculating that variable. If it is, I 
#    # return a METAL_Parameter node instead of Declaration node.
#    # So I need to confidently say that it is a computation for global thread index.
#    #cuda_vars = []
#    print(f"NODE: {node}")
#    #cuda_vars += cuda_vars
#    #while isinstance(node, (Binary, CudaVar)): # not working. Infinite loop
#    if isinstance(node, Binary): # node=Bin(op, left, right)
#        print("left: ", node.left)
#        left = check_semantic(node.left, cudaVars, node.op) # left=Bin(op, left, right)
#        print("left 2: ", left)
#        print("right: ", node.right)
#        right = check_semantic(node.right, cudaVars, node.op)
#        print("right 2: ", right)
#    elif isinstance(node, CudaVar):
#        print("it is CudaVar instance: ", node)
#        #cuda_vars.append(metal_map(node.base))
#        #cuda_vars.append(node.base + "." + node.dim)
#        var = node.base + "." + node.dim
#        cudaVars.append(var)
#    if op is not None:
#        cudaVars.append(op)
#
#    print("cuda_vars2: ", cudaVars)
#    return cudaVars


def check_semantic(node, expr, op=None):
    if isinstance(node, Binary):
        check_semantic(node.left, expr, node.op) if isnode(node.left) else None#print("oi")
        expr.append(node.op)
        check_semantic(node.right, expr, node.op) if isnode(node.right) else None# print("oi")
    elif isinstance(node, CudaVar):
        expr.append(node.base + "." + node.dim)
    return expr


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
