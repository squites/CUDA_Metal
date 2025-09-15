from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_IfStatement, METAL_ForStatement, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Program
from ast_builder import Parameter, Declaration, Assignment, Binary, Literal, CudaVar, Variable, Array

class CUDAVisitor(object):
    """ Traverse the ast nodes """
    def __init__(self):
        #self.buffer_idx = -1
        self.kernel_params = []
        self.body = []
        self.cudavarslist = []

    # passing the node parent
    def visit(self, node, parent=None, idx=0):
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
                param_idx += 1
                child_node = self.visit(child, idx=param_idx)
                if isinstance(child, Parameter):
                    self.kernel_params.append(child_node)
                else:
                    body.append(child_node) 
            return METAL_Kernel(qualifier, type, name, self.kernel_params, body)
        else:
            return METAL_Kernel(qualifier, type, name, [], [])

    def visit_Parameter(self, node, buffer_idx=0):
        mem_type = metal_map(node.mem_type)
        if node.mem_type == "device" and node.type == "int*" or node.type == "float*":
            #buffer_idx = self.kernel_params.index(node) #get_param_idx(self.kernel_params, node) #+= 1
            buffer = f"[[buffer({buffer_idx})]]"
        else:
            mem_type = "" #"constant"
            buffer = None # when its not a vector, matrix or tensor
        return METAL_Parameter(memory_type=mem_type, type=node.type, name=node.name, buffer=buffer)

    def visit_Body(self, node):
        if node.children():
            statements = []
            for child in node.children():
                # for each child, will return the respective METAL node. Ex: visit(child) = METAL_Declaration(type, name, value)
                child_node = self.visit(child) # visit(Declaration), visit(Assignment)
                # check if its Parameter() node for tid and gid
                statements.append(child_node) # this right? It is if child_node is being a METAL node returned by visit methods. Not right if child_node is CUDA node
            return METAL_Body(statements)
        else:
            return METAL_Body(node.statement)

    def visit_Statement(self, node):
        pass

    # obs: for tid and gid, we are generating the right Parameter() node, but they are still inside the 
    # body of the kernel. We need to have a way to move them into Parameters when they're thread index 
    # calculations.
    # obs2: if i set a self.cudavarslist = [], this will store every Declaration node cuda vars. If i use it inside the 
    # function only the local list, I'll have only the current node cuda vars. Which one is better?
    def visit_Declaration(self, node, parent=None):
        expr = extract_cuda_expression(node, expr="")
        print(f"expr: {expr}\n")
        cudavars = []
        #cudavars = ""
        memory = node.memory if node.memory else None
        type = node.type
        name = self.visit(node.name) if isnode(node.name) else node.name
        if node.children():
            value = [] # = Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                #print("\nchild ", child)
                cudavars = check_semantic_v2(child, cudavars, parent=node)
                self.cudavarslist.append(cudavars) if cudavars != [] else None
                #print(f"cudavars: {cudavars}\nlen: {len(cudavars)}")
                #print(f"self.cudavarslist: {self.cudavarslist}")

                # outadated
                # idea: if there's cuda var inside body, create METAL_Parameter node and add to self.kernel_params
                # then, check if that node already have been added. If not add to the list.

                # probably the stupidest way to make this, but that'll do for now
                for var in range(len(self.cudavarslist)):                                 # [var[0(0,)]]
                    node = None
                    # wrong! Im passing the name there, but i need to check the node
                    #if not check_nodes(self.cudavarslist[var][0][0], self.kernel_params): # [ [ ( , )]]
                    if not check_nodes(node, self.kernel_params):
                        #print(f"var: {self.cudavarslist[var][0][0]} ---  INSERE")
                        x = self.cudavarslist[var][0][0]
                        #node = METAL_Parameter(memory_type=None, type="uint3", name=x, buffer=None, init=metal_map(x))
                        #self.kernel_params.append(self.cudavarslist[var][0][0])
                        #self.kernel_params.append(node)
                
                # call function to map the vars
                node = METAL_Parameter(memory_type=None, type="uint3", name="blockidx", buffer=None, init=None)
                #if not check_nodes(node, self.kernel_params):
                #    self.kernel_params.append(node)
                #print("self.kernel_params: ", self.kernel_params)
                #return None # this was generating the error on for codegen, where it was concatenating None
                #return METAL_Parameter(memory_type=None, type="uint", name=name, buffer=None, init=param)
                child_node = self.visit(child, parent=node) # this is equal as: "return METAL_Declaration(type, name, value)"
                value.append(child_node)
            #print("self.kernel_params: ",  self.kernel_params)
            if value != None:
                return METAL_Declaration(metal_map(memory), type, name, value)
            else:
                return METAL_Declaration(metal_map(memory), type, name)
        else:
            return METAL_Declaration(metal_map(memory), type, name, value=node.value)
        

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
        #if node.init and node.condition and node.increment is not None:
        init = self.visit(node.init, parent=node) #if node.init is not None else ""
        cond = self.visit(node.condition, parent=node)
        incr = self.visit(node.increment, parent=node)
        stmts = []
        # for some reason when I print `node.children()`, I get a double list [[...]]
        for child in node.children()[0]:
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
        case "blockIdx":        metal_term = "[[threadgroup_position_in_grid]]"
        case "blockIdx.x":      metal_term = "[[threadgroup_position_in_grid]]"
        case "threadIdx":       metal_term = "[[thread_position_in_threadgroup]]"
        case "blockDim":        metal_term = "[[threads_per_threadgroup]]"
        case "__global__":      metal_term = "device" # using global memory
        case "__shared__":      metal_term = "threadgroup" # using shared memory
        case "__constant__":    metal_term = "constant"
        case "blockIdx.x * blockDim.x + threadIdx.x": metal_term = "[[thread_position_in_grid]]"
        case "blockIdx.y * blockDim.y + threadIdx.y": metal_term = "[[thread_position_in_threadgroup]]"
        case "blockIdx.y":     metal_term = "[[thread_position_in_threadgroup]]"
        case "__syncthreads()": metal_term = "threadgroud_barrier()"
    return metal_term

def map_names(name):
    match name:
        case "blockIdx":    return "tg_pig"
        case "threadIdx":   return "t_pit"
        case "blockDim":    return "t_pt"

def check_nodes(node, params):
    if node in params:
        return True
    else:
        return False

def get_param_idx(kernel_params, node):
    for p in kernel_params:
        if node.type == p.type and node.name == p.name:
            return int(p)
    return 0

def check_semantic_v2(node, cudalist, parent=None):
    """
    this is to group cudavars that we have in the cuda kernel, so we can translate them semantically the right way.
    """
    #if parent is not None:
    #    print("\nparent: ", parent.name)
    if isinstance(node, CudaVar):
        t = (node.base, node.dim)
        #t = str(node.base) + "." + str(node.dim) # maybe is better to return like this
        cudalist.append(t)
        #cudalist = t
    elif isinstance(node, Binary):
        check_semantic_v2(node.left, cudalist) if isnode(node.left) else None
        cudalist.append(node.op)
        #cudalist += str(node.op)
        check_semantic_v2(node.right, cudalist) if isnode(node.right) else None
    return cudalist # returning tuple
    #return [node, cudalist] # do i need to return the node and the expression?

# This produces the right cuda string based on the nodes! But we can't rely only on the string to compare to generate
# a semantic node. Instead what we want to do is to analyse the tree structure of that node, and based on the structure
# already be capable of knowning what that node represents. We don't need to check the string. Because we can calculate
# the same id with different position for example, and the string won't be the same. We want to do pattern_matching
def extract_cuda_expression(node, expr):
    print("node to analyse: ", node)
    if isinstance(node, Binary):
        expr = extract_cuda_expression(node.left, expr=expr) #if isnode(node.left) else None
        expr += f" {str(node.op)} "
        expr = extract_cuda_expression(node.right, expr=expr) #if isnode(node.right) else None
    elif isinstance(node, CudaVar):
        expr += f"{node.base}.{node.dim}"
    elif isinstance(node, Declaration) and node.value is not None:
        expr = extract_cuda_expression(node.value, expr)
    return expr

def semantic_analysis(expr):
    # analyse the cuda string extracted 
    pass

def pattern_matching(node):
    """ checks the structure of the sub-tree of this node, and based on that structure generate a semantic node """
    # this is the one that matters! figure it out a way to analyse the node subtree and check what computation is doing.
    # if its something as global thread id, create a new node for it.
    # should I use this for smem nodes as well? I dont think so
    assert(isinstance(node, (Declaration, Assignment)))
    print("node: ", node)
  

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
# --- Understanding memory
# if you parameter is: `device float* a`, then `a` points to GPU device memory. It is a buffer that must be allocated
# and bound from the CPU side.
# if it is: `constant float* a`, its also a buffer but immutable and read-only for the kernel.
#
# When we use `[[buffer(n)]]`?
# if the parameter is a device or constant pointer to memory, must have `[[buffer(i)]]`. If it comes from CPU as a 
# buffer (MTLBuffer or setBytes) -> mark it with [[buffer(i)]]
#
# How does the CPU and GPU communicate in Metal?
# there's a shared CPU/GPU memory state between them. It has multiple buffer levels inside this memory state.
# - CPU basically says: `setBuffer at index 0 with array A`. So stores the array A at buffer 0 on shared memory state 
# - GPU can read the values store in these buffers
# `device float* array [[buffer(2)]]`: this means that we can write to this buffer `array` at [[buffer(2)]]
# - `uint2 index [[thread_position_in_grid]]`: This means that we have `index` for the thread in a 2D grid.
# For example: in CUDA we write:
# `row = blockIdx.x * blockDim.x + threadIdx.x;` and 
# `col = blockIdx.y * blockDim.y + threadIdx.y;`
# In Metal we have:
# `uint2 index [[thread_position_in_grid]];`
# `row = index.x;`
# `col = index.y;`
# Example:
# x, x, x,
# x, x, x,
# T, x, x,
#
# T has position uint2 = (0,2) in the grid
#
#
# uint2 tid [[thread_position_in_threadgroup]] -> threadIdx.{x, y}
# uint2 gid [[threadgroup_position_in_grid]]   -> blockIdx.{x, y}
# uint2 tg_size [[threads_per_threadgroup]]    -> blockDim.{x, y}
#
#
#
# Obs: What I need to do:
# - generate parameter variables that represent threadIdx, one for blockIdx, and one for blockDim.
# So, if there's a use of threadIdx on the cuda code, create this parameter, if there's blockIdx being used on cuda code,
# generate this parameter instead, and so on.
# - then, we need to map which variable in cuda that's using which cuda variable, and use the Parameter variable node
# to make the computation. 
# Ex: 
# in cuda: int tRow = threadIdx.x / CHUNKSIZE;
#
# in metal we generate a metal parameter node `uint3 tid [[thread_position_in_grid]]`
# then on the computation we use: int `tRow = tid.x / CHUNKSIZE`. We need to map the parameter variable to use in 
# calculation 
# 
# !!!!!!!!!! IMPORTANT !!!!!!!!!!
# IMPORTANT: IMPLEMENT SEMANTIC ANALYSES. Detect what nodes are calculating. So for example, if I have a Binary node that computes
# blockIdx.x * blockDim.x + threadIdx.x; mark that node that it calculates linear thread ID in the block. We can do this
# by adding a flag to the node, or making new nodes