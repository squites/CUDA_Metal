from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_IfStatement, METAL_ForStatement, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Program
from ast_builder import Parameter, Declaration, Assignment, Binary, Literal, CudaVar, Variable, Array, StartBlockIdx

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
    # obs2: if i set a self.cudavarslist = [], this will store every Declaration node cuda vars. If i use it inside the 
    # function only the local list, I'll have only the current node cuda vars. Which one is better?
    def visit_Declaration(self, node, parent=None):
        #expr = extract_cuda_expression(node, expr="")
        semantic_analysis(node)
        #print(f"expr: {expr}\n")
        cudavars = []
        #cudavars = ""
        memory = node.memory if node.memory else None
        type = node.type
        name = self.visit(node.name) if isnode(node.name) else node.name
        if node.children():
            value = [] # = Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                #cudavars = check_semantic_v2(child, cudavars, parent=node)
                #self.cudavarslist.append(cudavars) if cudavars != [] else None

                # probably the stupidest way to make this, but that'll do for now
                #for var in range(len(self.cudavarslist)):                                 # [var[0(0,)]]
                #    node = None

                #    if not check_nodes(node, self.kernel_params):
                #        x = self.cudavarslist[var][0][0]
                
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


# need to add after a tree normalization
# so if we have a node like Add(a, Add(b, c)) transform into Add(a, b, c)

def semantic_analysis(node): # this calls pattern_matching, where it'll classify the child node based on the semantic meaning
    """ checks the structure of the sub-tree of this node, and based on that structure generate a semantic node """
    # this is the one that matters! figure it out a way to analyse the node subtree and check what computation is doing.
    # if its something as global thread id, create a new node for it.
    #
    # How the pattern matching must work:
    # 1 - canonicalize(normalize): operations stays the same format(order) always
    # 2 - pattern matching: when you see some specific node, rewrite to a new semantic node ex: Global1DThreadId()
    assert isinstance(node, (Declaration, Assignment, Binary, CudaVar)), "Invalid node!"
    print("-------------------------------------------------------------------------------------")
    print("SEMANTIC ANALYSIS:\n", node)
    for child in node.children():
        pattern_matching(child)
        # ...

def pattern_matching(node): # will recursively go down until there's no more Binary node, and return a pattern for it
    print("PATTERN MATCHING:\n", node)
    new_node = canonicalize(node)
    # try to normalize and find get the pattern matching here

def canonicalize(node): # here will rewrite the node changing the order of the factors, so they can always be the same
    print("CANONICALIZE:\n", node)
    ops = ["+", "*"]
    if isinstance(node, Binary) and node.op in ops:
        #terms = reorder(flatten(node, node.op))
        terms = flatten(node, node.op)  # flatten "+" terms
        #print("flatten +:", terms)
        for t in range(len(terms)):
            if isinstance(terms[t], Binary):    
                terms[t] = flatten(terms[t], terms[t].op) # flatten "*" factors

        print("flattened terms:", terms)
        # add tags for each term: e.g. (CudaVar(base="threadIdx", dim="x"), "thread")
        tag_terms = addtag(terms)
        print("tag terms:", tag_terms)
        reorder_terms = reorder(tag_terms)
        print("reorder terms:", reorder_terms)
        #rebuild(node, terms)#rebuild_node(node, ordered_terms)
        #compare(terms)
        # reconstruct node based on the ordered terms
    else: # tirar else
        return node

def flatten(node, op):
    """ separates the terms individually (in nodes). At first we separate by `+`, but then all commutative ops """
    #print(f"FLATTEN OP({op}):\n{node}") # use this to see which node is being flatted
    # if i take off the op == node.op, we flatten everything, but we don't separate the factors vs terms, so we have:
    # [1, tidx, bidx, bdim] instead of [[1, tidx], [bidx, bdim]]
    if isinstance(node, Binary) and op == node.op:
        left = flatten(node.left, op=node.op)
        right = flatten(node.right, op=node.op)
        return left+right
    else:
        return [node]

# [[Cuda("threadIdx")], [Cuda("blockIdx"), Cuda("blockDim")]]
# this code is disgusting! But was the initial way that worked
def addtag(terms): # maybe transform into a dict?
    print("ADD TAG:\n", terms)
    for term in range(len(terms)):
        if not isinstance(terms[term], list):
            terms[term] = [terms[term]]
        for sub in range(len(terms[term])):
            if isinstance(terms[term][sub], CudaVar):
                if "thread" in terms[term][sub].base:
                    terms[term][sub] = (terms[term][sub], "thread")
                elif "block" in terms[term][sub].base:
                    terms[term][sub] = (terms[term][sub], "block")
                else:
                    terms[term][sub] = (terms[term][sub], "grid")
            elif isinstance(terms[term][sub], Literal):
                terms[term][sub] = (terms[term][sub], "literal")

    return terms


# refactor this eventually, to get to call flattened only once instead of twice
# PROBLEM: we are only reordering the subterms. But we are not ordering the terms. So if we have:
# blockDim.x * blockIdx.x + threadIdx.x -> will be: [[blockIdx.x, blockDim.x], threadIdx.x], and it should be:
#                                                   [threadIdx.x, [blockIdx.x, blockDim.x]]
# possible solution: reorder based on the size of the subterm as well. If len(subterm) is 1, go to left.
# solution: flatten everything first, and then reorder. Need to change swap function probably and also reorder. 
# first reorder the inner flattened terms (*) and then the outer terms (+)

def reorder(terms):
    print("REORDER:\n", terms)
    
    #for term in terms:
    #    term = swap(term, terms) # inner swap first
    terms = swap(terms)
        #if isinstance(term, list):
        #    term = swap(term) # this swap each term
    print("terms:", terms)
        #for subterm in term:


#def reorder(terms):
#    """ get the flattened terms separated by `*` and reorder the nodes based on priority """
#    print("REORDER 2:\n", terms)
#    # [(1 * threadIdx.x), (blockIdx.x * blockDim.x)] -> len=2
#    # needs to be in the end:
#    # [[1, threadIdx.x], [blockIdx.x, blockDim.x]] -> len=2x2
#
#    for t in range(len(terms)):
#        if isinstance(terms[t], Binary):# and terms[t].op == "*":
#            op = terms[t].op
#            terms[t] = flatten(terms[t], op=terms[t].op)
#            print("terms[t] flatten: ", terms[t])
#            #terms[t] = swap(terms[t], op=op)
#            #print("terms[t] swap: ", terms[t])        
#    return terms

# Maybe we need to change this. After we flatten the terms/factors, we may translate each element to what category
# that element is. So, if we have 'threadIdx.x', we translate into "thread". If we have blockIdx.x or blockDim.x
# we translate to "block". This way, instead of having [[threadIdx.x], [blockIdx.x, blockDim.x]] we'll have
#[[thread], [block, block]].
# Doing this we abstract, so instead of this saying: "this was blockIdx.x * blockDim.x specifically", this now means:
# "this expression had a product of 2 block-level terms".
#
# To do that, we keep both layers of info. (before flatten):
# node: CudaVar(base="threadIdx", dim="x"), and
# semantic tag: (category="thread")
# Then we flatten and reorder based on the semantic tag (for canonicalization)
# 
# Then on semantic node inference, once we detect a canonical signature pattern like: [[thread], [block,block]]
# we map semantically to GlobalThreadID
#def swap(terms, op="*"):
#    """ swap order of terms based on the hierarchy tidx->bidx->bdim->gdim """
#    print("SWAP:\n", terms)
#    hierarchy = {
#        "thread": 0,
#        "block": 1,
#        "grid": 2,
#        "literal": 3,
#        "variable": 4,
#    }
#    for i in range(len(terms)-1):
#        for j in range(1, len(terms)):
#            if isinstance(terms[i], CudaVar) and isinstance(terms[j], CudaVar):
#                if hierarchy.get(terms[i].base) > hierarchy.get(terms[j].base):
#                    tmp = terms[i]
#                    terms[i] = terms[j]
#                    terms[j] = tmp
#            elif (isinstance(terms[i], Literal) and not isinstance(terms[j], Literal)):
#                # (1, blockIdx) -> (blockIdx, 1)
#                tmp = terms[i]
#                terms[i] = terms[j]
#                terms[j] = tmp
#                # constant fold to remove the constants if they're unecessary
#                fold(terms, op)
#    return terms

def swap(terms): # used to be swap(term, terms):
    """ swap the order of inner and outer terms based on the priority order (low->high) """
    print("SWAP:\n", terms)
    order = {    # [block, block]
        "thread": 0,
        "block": 1,
        "grid": 2,
        "literal": 3,
    }

    # v2 (inner sort)
    for t in range(len(terms)): # terms[t] == term
        for i in range(len(terms[t])-1):
            # fold here?
            for j in range(1, len(terms[t])):
                if isinstance(terms[t], list) and len(terms[t]) > 1:
                    if order.get(terms[t][i][1]) > order.get(terms[t][j][1]):
                        tmp = terms[t][i]
                        terms[t][i] = terms[t][j]
                        terms[t][j] = tmp
        # fold here?
        # ... here fold will only work with '*' expr, because each 't' term is a mul expr
    print("inner sorted terms:", terms)

    # (outer sort)
    for t1 in range(len(terms)-1):
        print("terms[t1]:", terms[t1])
        for t2 in range(1, len(terms)):
            print("terms[t2]:", terms[t2])
            # since its already inner sorted, the first tag will always be the priority tag of that term
            if order.get(terms[t1][0][1]) > order.get(terms[t2][0][1]):
                tmp = terms[t1]
                terms[t1] = terms[t2]
                terms[t2] = tmp

    print("outer sorted terms:", terms)

    return terms


def fold(terms, op="*"):
    print("FOLD:\n", terms, op)
    for sub in range(len(terms)):
        if isinstance(terms[sub], Literal):
            if terms[sub].value == "1" and op == "*":
                terms.remove(terms[sub])
            elif terms[sub].value == "0" and op == "+":
                terms.remove(terms[sub])
            elif terms[sub].value == "0" and op == "*":
                return None
            else:
                # accumulate constants to be one (not implemented yet!)
                #acc += terms[sub].value
                pass
    return terms








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








# !!!!!!!!!! IMPORTANT !!!!!!!!!!
# IMPORTANT: IMPLEMENT SEMANTIC ANALYSES. Detect what nodes are calculating. So for example, if I have a Binary node that computes
# blockIdx.x * blockDim.x + threadIdx.x; mark that node that it calculates linear thread ID in the block. We can do this
# by adding a flag to the node, or making new nodes