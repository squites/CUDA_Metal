#from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_IfStatement, METAL_ForStatement, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Program
#from ast_builder import Parameter, Declaration, Assignment, Binary, Literal, CudaVar, Variable, Array, ThreadIdx, BlockIdx, BlockDim, GlobalThreadIdx, Mul, Add
from ast_builder import *

class CUDAVisitor(object):
    """ Traverse the ast nodes """
    def __init__(self):
        #self.buffer_idx = -1
        self.kernel_params = []
        self.body = []

    # passing the node parent
    def visit(self, node, parent=None, idx=0):
        method = "visit_" + node.__class__.__name__ # start: visit_CUDA_Program()
        visitor = getattr(self, method, self.visit_error) # visitor = self.visit_CUDA_Program()
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
        mem_type = metal_map(node.mem_type) # mem_type="device" when there's data pointer to GPU. So it's data stored on global memory, we need device 
        if node.mem_type == "device" and node.type == "int*" or node.type == "float*":
            buffer = buffer_idx
        else:
            mem_type = None 
            buffer = None
        return METAL_Parameter(memory_type=mem_type, type=node.type, name=node.name, attr=None, buffer=buffer, init=None)

    def visit_Body(self, node):
        if node.children():
            statements = []
            for child in node.children():
                # for each child, will return the respective METAL node. Ex: visit(child) = METAL_Declaration(type, name, value)
                child_node = self.visit(child) # visit(Declaration), visit(Assignment)
                print("child node: ", child_node)
                # check if its Parameter() node for tid and gid
                if child_node is not None: # this filters in cases like GlobalThreadIdx, because they're added to params, so there's no need to add them in body
                    statements.append(child_node) 
            return METAL_Body(statements)
        else:
            return METAL_Body(node.statement)

    def visit_Declaration(self, node, parent=None):
        print("\n\nVISIT_DECLARATION:\n", node)
        node = pattern_matcher(node) # when we rewrite the IR with GlobalThreadIdx() node for example, the code calls the visit_error() function, because there's no visit_GlobalThreadIdx() node for it. This could be a problem when creating METAL_ast. Fix that later!
        memory = node.memory if node.memory else None
        type = node.type
        name = self.visit(node.name) if isnode(node.name) else node.name

        if isinstance(node.value, GlobalThreadIdx):
            param = METAL_Parameter(memory_type=None, type="uint", name=node.name, attr="thread_position_in_grid", buffer=None, init=None)
            if not check_param(self.kernel_params, param.attr): # to not add repetitive vars on params
                self.kernel_params.append(param)
            return None # dont return node because already added to kernel_params

        if node.children():
            value = [] # = Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                print("child: ", child)
                child_node = self.visit(child, parent=node) # this makes the METAL AST node to have METAL_Binary instead of Binary, for Declarations that have Binary nodes as values for example.
                print("child_node: ", child_node)
                value.append(child_node)
            print("values: ", value)

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
        init = self.visit(node.init, parent=node)
        cond = self.visit(node.condition, parent=node)
        incr = self.visit(node.increment, parent=node)
        stmts = []
        # for some reason when I print `node.children()`, I get a double list [[...]]
        for child in node.children()[0]:
            child_node = self.visit(child, parent=node)
            stmts.append(child_node)
        return METAL_ForStatement(init=init, condition=cond, increment=incr, forBody=stmts, parent=parent)

    def visit_Binary(self, node, parent=None):
        #self.parent = node if parent is not None else None
        #metal_op = node.op
        #left = self.visit(node.left, parent=node) if isnode(node.left) else str(node.left)
        #right = self.visit(node.right, parent=node) if isnode(node.right) else str(node.right)
        metal_op = node.op
        print("node.left: ", node.left)
        if isnode(node.left):
            print("ISNODE")
            left = self.visit(node.left, parent=node)
        elif node.left.isdigit():
            print("ISDIGIT")
            left = METAL_Literal(node.left)
        else:
            print("ISVAR")
            left = METAL_Variable(node.left)
        print("LEFT:", left)

        if isnode(node.right):
            right = self.visit(node.right, parent=node)
        elif node.right.isdigit():
            right = METAL_Literal(node.right)
        else:
            right = METAL_Variable(node.right)
        
        return  METAL_Binary(metal_op, left, right)

    def visit_Literal(self, node, parent=None):
        value = node.value
        return METAL_Literal(value=value)

    def visit_Variable(self, node, parent=None):
        #name = node.name
        return METAL_Variable(node.name)

    def visit_Array(self, node, parent=None):
        array_name = self.visit(node.name, parent=node) if isnode(node.name) else node.name
        idx = self.visit(node.index)
        return METAL_Array(array_name, idx)

    def visit_CudaVar(self, node, parent=None):
        metal_var = metal_map(node.base)
        return METAL_Var(metal_var)

    # visit IR nodes. OBS: Metal doesn't have built-ins so all cudavar must be passed as argument to metal kernel
    def visit_Add(self, node, parent=None):
        operands = [self.visit(op) for op in node.operands]
        res = operands[0]
        for op in operands[1:]:
            res = METAL_Binary("+", res, op)
        return res

    def visit_Mul(self, node, parent=None):
        operands = [self.visit(op) for op in node.operands]  
        res = operands[0]
        for op in operands[1:]:
            res = METAL_Binary("*", res, op)
        return res
    
    def visit_ThreadIdx(self, node, parent=None):
        name = "tidx"
        if not check_param(self.kernel_params, "thread_index_in_threadgroup"):
            param = METAL_Parameter(memory_type=None, type="uint", name=name, attr="thread_index_in_threadgroup", buffer=None, init=None)
            self.kernel_params.append(param)
        return METAL_Variable(name=f"{name}.{node.dim}") # it was: name=f"threadIdx.{node.dim}"

    def visit_BlockIdx(self, node, parent=None):
        name = "tgpos"
        if not check_param(self.kernel_params, "threadgroup_position_in_grid"):
            param = METAL_Parameter(memory_type=None, type="uint", name=name, attr="threadgroup_position_in_grid", buffer=None, init=None)
            self.kernel_params.append(param)
        return METAL_Variable(name=f"{name}.{node.dim}") # it was: name=f"blockIdx.{node.dim}"
    
    def visit_BlockDim(self, node, parent=None):
        name = "tptg"
        if not check_param(self.kernel_params, "threads_per_threadgroup"):
            param = METAL_Parameter(memory_type=None, type="uint", name=name, attr=f"threads_per_threadgroup", buffer=None, init=None)
            self.kernel_params.append(param)
        return METAL_Variable(name=f"{name}.{node.dim}") #it was: f"blockDim.{node.dim}"
    
    # error call function
    def visit_error(self, node, attr): 
        print(f"The node {node} has no attribute named {attr}!")

def check_param(params, attr):
    for p in params:
        if p.attr == attr:
            return True
    return False

# helpers (move this to another file)
def isnode(node):
    """ Check if the node that we're visiting has any node as value for any attribute """
    return isinstance(node, (Binary, Literal, Variable, Array, CudaVar, ThreadIdx, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Var))

# i guess i can remove stuff from here, like: "blockIdx.x * blockDim.x + threadIdx.x": metal_term = "[[thread_position_in_grid]]" 
def metal_map(cuda_term):
    """ Maps CUDA concept syntax into METAL concept syntax"""
    metal_term = None
    match cuda_term:
        case "blockIdx":        metal_term = "[[threadgroup_position_in_grid]]"
        case "blockIdx.x":      metal_term = "[[threadgroup_position_in_grid]]"
        case "threadIdx":       metal_term = "[[thread_position_in_threadgroup]]"
        case "blockDim":        metal_term = "[[threads_per_threadgroup]]"
        case "__global__":      metal_term = "device" # using global memory
        case "__shared__":      metal_term = "threadgroup" # using shared memory
        case "__constant__":    metal_term = "constant"
        case "blockIdx.x * blockDim.x + threadIdx.x": metal_term = "[[thread_position_in_grid]]"
        case "blockIdx.y * blockDim.y + threadIdx.y": metal_term = "[[thread_position_in_threadgroup]]" # add this new rule
        case "blockIdx.y":     metal_term = "[[thread_position_in_threadgroup]]"
        case "__syncthreads()": metal_term = "threadgroud_barrier()"
    return metal_term

# ---------------------------------------------------------------------------
#
# THIS FILE SHOULD END HERE!!!! EVERYTHING BELOW MUST BE MOVED SOMEWHERE ELSE
#
# ---------------------------------------------------------------------------

def pattern_matcher(node):
    assert isinstance(node, (Declaration, Assignment, Binary, CudaVar)), "Invalid node!"
    print("-------------------------------------------------------------------------------------")
    print("PATTERN MATCHING:\n", node)
    for child in node.children():
        ir = lowering(child)
        print(" LOWERED IR: ", ir)
        ir = canonicalize(ir)
        print(" CANONICALIZED IR: ", ir)
        if ir is not None:
            ir = IRrewrite(ir)
        
        node.value = ir
        print("NEW NODE VALUE:\n", node)
    return node

# create this function?
def lowering(node):
    print("LOWERING:\n", node)
    ops = ["+", "*"]
    terms = None
    if isinstance(node, Binary) and node.op in ops:
        terms = flatten(node, node.op)
        for t in range(len(terms)):
            terms[t] = flatten(terms[t], terms[t].op) if isinstance(terms[t], Binary) else [terms[t]]
        
        print("FLATTENED:\n", terms)
        terms = IRconstruct(terms)
        #return terms

    # added for %, /, cases
    elif isinstance(node, Binary):
        # lower children
        left = lowering(node.left) if isinstance(node.left, (Binary, CudaVar)) else node.left
        right = lowering(node.right) if isinstance(node.right, (Binary, CudaVar)) else node.right
        return Binary(op=node.op, left=left, right=right)

    elif isinstance(node, CudaVar):
        terms = IRconstruct(node)

    else:
        terms = node
    print("IR CONSTRUCTED:\n", terms)
    return terms    

# OBS: moved the flatten/IRconstruct out of canonicalize to lowering()!
def canonicalize(terms): # here will rewrite the node changing the order of the factors, so they can always be the same
    print("CANONICALIZE:\n", terms) # Add(...)
    if isinstance(terms, Add):
        terms = reorder(terms)
        print("REORDERED: ", terms)
        return terms
    return terms # this is for nodes that aren't Declaration(Bin). not working yet

# keep flatten the way it is. The rewrite will be after, changing [] by Mul() and Add() IR nodes. This process is called
def flatten(node, op="*"):
    """ separates terms individually (in nodes). At first we separate by `+`, but then all commutative ops """
    if isinstance(node, Binary) and op == node.op:
        left = flatten(node.left, op=node.op)
        right = flatten(node.right, op=node.op)
        return left+right
    else:
        return [node]

def taglvl(node):
    tag = None
    if isinstance(node, ThreadIdx): tag = "thread"
    elif isinstance(node, (BlockIdx, BlockDim)): tag = "block"
    elif isinstance(node, Literal): tag = "literal"
    else: tag = "grid"
    print("TAGLVL: ", node, tag)

    return tag

def reorder(node):
    print("REORDER:\n", node)
    order = {
        "thread": 0,
        "block": 1,
        "grid": 2,
        "literal": 3,
    }
    # inner sort
    for mul in node.operands:
        print("mul:", mul)
        mul.operands = sorted(mul.operands, key=lambda x: order.get(taglvl(x), 99))
        print("reordered mul:", mul)#.operands)

        # fold
        for i in mul.operands:
            #print("i: ", i)
            if isinstance(i, Literal):
                fold(mul, op="*")
                print(f"folded: {mul}")
                break # jumps outside the for i in mul.operands loop

    # outer sort
    node.operands = sorted(node.operands, key=lambda m: order.get(taglvl(m.operands[0]), 99))
    return node

# new fold version
def fold(terms, op="*"):
    print("FOLD: \n", terms)
    assert isinstance(terms, Mul), "Wrong object!"
    acc = 1 if op == "*" else 0
    # keep track of the node types (maybe add a unique loop, and appends on each one)
    literals = [sub for sub in terms.operands if isinstance(sub, Literal)]
    vars = [sub for sub in terms.operands if not isinstance(sub, Literal)]
    print("literals: ", literals)
    print("vars: ", vars)

    for lit in literals:
        acc = acc*int(lit.value) if op == "*" else acc+int(lit.value)
        print("acc:", acc)

    if acc == 1 and op=="*":
        terms.operands = vars
    elif acc == 0:
        pass
    else:
        terms.operands = vars + [Literal(value=acc)]

    print("RETURNED: ", terms)
    #return terms

def lower_cuda(node):
    if isinstance(node, CudaVar):
        if node.base == "threadIdx": return ThreadIdx(dim=node.dim) 
        elif node.base == "blockIdx": return BlockIdx(dim=node.dim)
        elif node.base == "blockDim": return BlockDim(dim=node.dim)
    return node

# adds Mul() and Add() IR nodes. Takes the ordered canonical flattened expr and rewrite with Mul() and Add() nodes.
def IRconstruct(expr):
    # single node
    if isinstance(expr, CudaVar):
        return lower_cuda(expr)

    # flattend list
    if isinstance(expr, list):
        for inner in range(len(expr)):
            expr[inner] = [lower_cuda(x) for x in expr[inner]]
            expr[inner] = Mul(expr[inner])
        return Add(expr)

    return expr


# adding high-level semantic nodes to the expressions
# Add(operands=[Mul(operands=[ThreadIdx('x')]), Mul(operands=[BlockIdx('x'), BlockDim('x')])]) -> GlobalThreadIdx()
# Move this to new .py file!
class Rule:
    def __init__(self, name, fpattern, fbuilder):
        self.name = name
        self.fpattern = fpattern
        self.fbuilder = fbuilder
    
    def match(self, node):
        print("matching: ", node)
        binds = self.fpattern(node)
        if binds is not None:
            return self.fbuilder(binds)
        return None

class Rewriter:
    def __init__(self, rules):
        self.rules = rules

    def rewrite(self, node):
        print("Rewriting... ", node)
        if not hasattr(node, "operands"):
            print("LEAF!")
            return node
        nodeops = [self.rewrite(child) for child in node.operands]
        print("nodeops: ", nodeops)

        for rule in self.rules:
            print("RULE: ", rule.name)
            x = rule.match(node)
            print("x: ", x)
            if x is not None:
                print("MATCH!")
                return x
        print("NO MATCH!")
        return node

# pattern functions:
def pat_GlobalThreadIdx(node):
    print("Pattern function: ", node)
    if not isinstance(node, Add):
        return None
    if len(node.operands) != 2:
        return None
    l, r = node.operands
    print(f"l {l}\nr {r}")
    if not isinstance(l, Mul) or not isinstance(r, Mul):
        return None

    def isthread(x):
        return (len(x.operands) == 1 and isinstance(x.operands[0], ThreadIdx))

    def isblock(x):
        if len(x.operands) != 2:
            return False
        a,b = x.operands
        return (isinstance(a, BlockIdx) and isinstance(b, BlockDim)) or (isinstance(a, BlockDim) and isinstance(b, BlockIdx))

    if isthread(l) and isblock(r):
        return {
            "dim": l.operands[0].dim
        }
    else:
        return None

# builder functions:
def build_GlobalThreadIdx(binds):
    """ create node GlobalThreadIdx(params) """     
    print("Building Node: ", binds)
    return GlobalThreadIdx(dim=binds["dim"])

# calls Rewriter
def IRrewrite(subtree):
    print("IR subtree: ", subtree)
    # rules
    rule1 = Rule(
        "GlobalThreadIdx",    # name
        pat_GlobalThreadIdx,  # pattern function
        build_GlobalThreadIdx # builder function
    )

    rewriter = Rewriter([rule1])
    new_tree = rewriter.rewrite(subtree)
    print("RESULT: ", new_tree)
    return new_tree


# OBS:
# VERY IMPORTANT! In metal there are no built-in variables like in cuda. So we ALWAYS have to pass them as arguments
# to the kernel. Even when using a declaration like: `int a = threadIdx.x / BLOCKSZ;` In metal we need to pass as arg:
# `(uint tx [[thread_index_in_threadgroup]])` and then inside the kernel we do: `int a = tx / BLOCKSZ`
#
#
# BUGS:
# 1- (DONE!) When declarating a new variable like this: `int tCol = tidx.x`, we need to create a Declaration node
# where tCol is now a METAL_Variable. So the node should be: METAL_Declaration(..., name=METAL_Variable('tCol))
# e.g.: METAL_Declaration(memory=None, type='int', name='tCol', value=[METAL_Variable(name='tidx.x')])
#               ...
# And on later uses of that variable in the code, should be referenced as METAL_Variable, and not just the string 
# name.
# (wrong!)      
#   METAL_Assignment(name=METAL_Array(name='data0_s', index=METAL_Binary(op='+', 
#                               left=METAL_Binary(op='*', left='tRow', right='CHUNKSIZE'), right='tCol')...
# (right!)
#   METAL_Assignment(name=METAL_Array(name='data0_s', index=METAL_Binary(op='+', 
#                               left=METAL_Binary(op='*', left=METAL_Variable('tRow'), right='CHUNKSIZE'), 
#                               right=METAL_Variable('tCol')).
#
# TODO:
# optimizations:
# - improve `fold` function
# - algebraic simplification
# - fuse kernels. Add a 'opt' flag for user if he wants to fuse kernels

# make this canonicalization as a class eventually
#class ExprCanonicalizer:
#    """ canonicalize/normalize the expression to always have the same format """
#    def canonicalize(self, node):
#        if isinstance(node, Binary):
#            return self._canonicalize_bin(node)
#        
#    def _canonicalize_bin(self, node):
#        # node: Binary()
#        left = self.canonicalize(node.left) if isinstance(node.left, Binary) else node
#        right = self.canonicalize(node.right) if isinstance(node.right, Binary) else node
#        print("left: ", left)
#        print("right: ", right)