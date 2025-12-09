from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_IfStatement, METAL_ForStatement, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Program
from ast_builder import Parameter, Declaration, Assignment, Binary, Literal, CudaVar, Variable, Array, ThreadIdx, BlockIdx, BlockDim, GlobalThreadIdx, Mul, Add

class CUDAVisitor(object):
    """ Traverse the ast nodes """
    def __init__(self):
        #self.buffer_idx = -1
        self.kernel_params = []
        self.body = []
        self.cudavarslist = [] # remove

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
            buffer = f"[[buffer({buffer_idx})]]"
        else:
            mem_type = ""
            buffer = None
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

    # obs: for tid and gid, we are generating the right Parameter() node, but they are still inside the 
    # body of the kernel. We need to have a way to move them into Parameters when they're thread index 
    # calculations.
    def visit_Declaration(self, node, parent=None):
        semantic_analysis(node)
        memory = node.memory if node.memory else None
        type = node.type
        name = self.visit(node.name) if isnode(node.name) else node.name
        if node.children():
            value = [] # = Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                # call function to map the vars
                node = METAL_Parameter(memory_type=None, type="uint3", name="blockidx", buffer=None, init=None)
                child_node = self.visit(child, parent=node) # this is equal as: "return METAL_Declaration(type, name, value)"
                value.append(child_node)

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


def semantic_analysis(node): # this calls pattern_matching, where it'll classify the child node based on the semantic meaning
    """ checks the structure of the sub-tree of this node, and based on that structure generate a semantic node """
    # if its something as global thread id, create a new node for it.
    # How the pattern matching must work:
    # 1 - canonicalize(normalize): operations stays the same format(order) always
    # 2 - pattern matching: when you see some specific node, rewrite to a new semantic node ex: Global1DThreadId()
    assert isinstance(node, (Declaration, Assignment, Binary, CudaVar)), "Invalid node!"
    print("-------------------------------------------------------------------------------------")
    print("SEMANTIC ANALYSIS:\n", node)
    for child in node.children():
        # canonicalize()? 
        pattern_matching(child)
        # ...


def pattern_matching(node): # will recursively go down until there's no more Binary node, and return a pattern for it
    print("PATTERN MATCHING:\n", node)
    canonical_expr = canonicalize(node)
    print(" ** Canonical expr:", canonical_expr)
    # IR construct
    IR_construct(canonical_expr)
    # IR rewrite
    recognition(canonical_expr)
    print("canonical expr:", canonical_expr)
    #print("Recognition:", recog)
    # try to normalize and find get the pattern matching here
    # build_node(node, canonical_expr)
    # ....


def canonicalize(node): # here will rewrite the node changing the order of the factors, so they can always be the same
    print("CANONICALIZE:\n", node)
    ops = ["+", "*"]
    if isinstance(node, Binary) and node.op in ops:
        terms = flatten(node, node.op)  # flatten by "+" into terms
        for t in range(len(terms)):
            if isinstance(terms[t], Binary):    
                terms[t] = flatten(terms[t], terms[t].op) # flatten by "*" into factors

        print(" ** flattened:", terms)
        tagged = addtag(terms)
        print(" ** tagged:", tagged)
        reordered = reorder(tagged) # trying this version using .tag node attribute
        print(" ** reordered:", reordered)
        return reordered
    #return node # this is for nodes that aren't Declaration(Bin) # not working yet

# keep flatten the way it is. The rewrite will be after, changing [] by Mul() and Add() IR nodes. This process is called
# IR construction
def flatten(node, op):
    print(f"FLATTEN: {node} {op}")
    """ separates the terms individually (in nodes). At first we separate by `+`, but then all commutative ops """
    if isinstance(node, Binary) and op == node.op:
        left = flatten(node.left, op=node.op)
        right = flatten(node.right, op=node.op)
        return left+right
    else:
        return [node]

# addtag but directly on the node attr instead of creating a tuple. adding one more dim of complexity is bad
def addtag(terms):
    """ adds a tag on each node to represent what level that node works with (e.g. thread, block, grid, ...) """
    print("ADD TAG EXPR:\n", terms)
    for term in range(len(terms)):
        if not isinstance(terms[term], list):
            terms[term] = [terms[term]]
        for sub in range(len(terms[term])):
            if isinstance(terms[term][sub], CudaVar):
                if "thread" in terms[term][sub].base: 
                    terms[term][sub].tag = "thread"
                elif "block" in terms[term][sub].base: 
                    terms[term][sub].tag = "block"
                else: 
                    terms[term][sub].tag = "grid"
            elif isinstance(terms[term][sub], Literal):
                terms[term][sub].tag = "literal"
    #print("ADD TAG EXPR:\n", terms)
    return terms


def reorder(terms):
    """ reorders the terms based on the tag of each node """
    print("REORDER ATTR:\n", terms)
    order = {
        "thread":  0,
        "block":   1,
        "grid":    2,
        "literal": 3,
    }

    # (inner sort)
    for t in range(len(terms)): # terms[t] == term
        for i in range(len(terms[t])-1): # loop through factors of one term
            for j in range(1, len(terms[t])):
                if isinstance(terms[t], list) and len(terms[t]) > 1:
                    if order.get(terms[t][i].tag) > order.get(terms[t][j].tag):
                        tmp = terms[t][i]
                        terms[t][i] = terms[t][j]
                        terms[t][j] = tmp

        # fold
        for i in terms[t]:
            if isinstance(i, Literal):
                terms[t] = fold(terms[t], op="*") # need to figure it out how to pass the 'op' to 'fold()'
                print("returned fold: ", terms[t]) # debug
                break # added this break to fix terms[t] problem. Not sure if this is right, but it works
    print("inner sorted terms:", terms)

    # (outer sort)
    for t1 in range(len(terms)-1):
        for t2 in range(1, len(terms)):
            # since its already inner sorted, the first tag will always be the priority tag of that term (sublist)
            if order.get(terms[t1][0].tag) > order.get(terms[t2][0].tag):
                tmp = terms[t1]
                terms[t1] = terms[t2]
                terms[t2] = tmp
    print("outer sorted terms:", terms)
    return terms


def fold(terms, op="*"):
    """ constant folding to optimize the METAL ast """
    print("FOLD_ATTR:\n", terms)
    assert isinstance(terms, list), "terms to be folded are not list!"
    folded = terms.copy()
    remove = []
    node = None
    acc = 0 if op == "+" else 1

    for sub in range(len(terms)):
        if isinstance(terms[sub], Literal):
            if terms[sub].value == "1" and op == "*":
                folded.remove(folded[sub])
            elif terms[sub].value == "0" and op == "*":
                pass
            else:
                print(" ** accumulate **")
                acc = acc * int(terms[sub].value) if op == "*" else acc + int(terms[sub].value)
                node = terms[sub]
                remove.append(terms[sub])
        else:
            continue
    
    if remove != []:
        node.value = acc
        for i in remove:
            if i in folded:
                folded.remove(i)
        folded.append(node)
    return folded


# this should be the function that introduces Mul() and Add() IR nodes. Takes the ordered canonical flattened expr and
# rewrite with Mul() and Add() nodes.
# Actually, this has to be called BEFORE we rewrite with semantic nodes! 
def IR_construct(canonical_expr):
    print("IR construct:\n", canonical_expr)
    # if [[2 terms]] return MUL
    

# this not working! This list approach is probably wrong.
# This is IR rewrite. This is the last thing to be called.
def recognition(canonical_expr):
    print("RECOGNITION: \n", canonical_expr)
    node = None
    if canonical_expr is not None:
        for t in canonical_expr:
            for x in range(len(t)):
                if isinstance(t[x], CudaVar):
                    if t[x].base == "threadIdx": node = ThreadIdx(dim=t[x].dim)
                    elif t[x].base == "blockIdx": node = BlockIdx(dim=t[x].dim)
                    elif t[x].base == "blockDim": node = BlockDim(dim=t[x].dim)
                    t[x] = node
                
            # need to change in canonical_expr        
        print(canonical_expr)






def build_node(src_node, canonical_expr):
    pass


# TODO:
# right approach:
# 1- create atomic semantic nodes based on the list canonicalized. So ThreadIdx(dim=x), BlockIdx(dim=x), ...
#    replace the nodes on that list for these atomic ones. Even if the sublist means something, only replace to atomic nodes first.
# 2- pattern/rewrite (composition): rule engine to match the atomic nodes configuration into a semantic node:
#    so like: [ThreadIdx(dim=x), [BlockIdx(dim=x), BlockDim(dim=x)]] into -> GlobalThreadIdx(dim=x)
# 3- These composite semantic nodes will be part of the new METAL ast
# 4- lower the metal ast and codegen
# 
# Question: 1) should I replace the list [[a,b], [c,d]] where subterms means the terms are being multiplied, and outerterms
# means they're being added, and change that to nodes like Add() and Mul()?
#
# optimizations:
# - improve `fold` function
# - algebraic simplification
# - fuse small kernels (on forward pass there will be many kernels) obs: even if its not fused on cuda, if we can fuse on metal, we do that! 
# - add tree normalization (optimization). ex: Add(a, Add(b, c)) transform into Add(a, b, c)
# - transform into a new class, check below

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