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
        semantic_analysis(node) # when we rewrite the IR with GlobalThreadIdx() node for example, the code calls the visit_error() function, because there's no visit_GlobalThreadIdx() node for it. This could be a problem when creating METAL_ast. Fix that later!
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

# i guess i can remove stuff from here, like: "blockIdx.x * blockDim.x + threadIdx.x": metal_term = "[[thread_position_in_grid]]" 
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

# remove?
#def get_param_idx(kernel_params, node):
#    for p in kernel_params:
#        if node.type == p.type and node.name == p.name:
#            return int(p)
#    return 0

# ---------------------------------------------------------------------------
#
# THIS FILE SHOULD END HERE!!!! EVERYTHING BELOW MUST BE MOVED SOMEWHERE ELSE
#
# ---------------------------------------------------------------------------

# move to new file `pattern_match.py`
def semantic_analysis(node): # need this? or can remove all to pattern matching
    """ checks the structure of the sub-tree of this node, and based on that structure generate a semantic node """
    # if its something as global thread id, create a new node for it and replace it.
    # 1 - canonicalize
    # 2 - pattern matching: rewrite to a new semantic node ex: GlobalThreadIdx()
    assert isinstance(node, (Declaration, Assignment, Binary, CudaVar)), "Invalid node!"
    print("-------------------------------------------------------------------------------------")
    print("SEMANTIC ANALYSIS:\n", node) # Declaration(...) essentially
    for child in node.children():
        print("child:", child)
        # canonicalize()? 
        #child_val = pattern_matching(child)
        #node.value = child_val
        node.value = pattern_matching(child)
    print("NEW NODE VALUE:\n", node)


def pattern_matching(node): # will recursively go down until there's no more Binary node, and return a pattern for it
    print("PATTERN MATCHING:\n", node) # Binary(...)
    canon = canonicalize(node)
    print(" ** Canonical expr:", canon) # canonicalized and folded
    # IR construct
    if canon is not None:
        newIR = IRconstruct(canon)
        print("IR: ", newIR)
        newIR = IRrewrite(newIR)
        print("NODE: ", node)
        print("FINAL IR: ", newIR)
        return newIR # just checking if Declaration node value will now change to GlobalThreadIdx
    return None


def canonicalize(node): # here will rewrite the node changing the order of the factors, so they can always be the same
    print("CANONICALIZE:\n", node) # Binary(...)
    ops = ["+", "*"]
    if isinstance(node, Binary) and node.op in ops:
        terms = flatten(node, node.op)  # flatten by "+" into terms
        for t in range(len(terms)):
            terms[t] = flatten(terms[t], terms[t].op) if isinstance(terms[t], Binary) else [terms[t]]

        # call IR construct
        terms = IRconstruct(terms)

        print(" ** flattened:", terms)
        terms = addtag(terms)
        print("tagged: ", terms)
        terms = reorder(terms)
        print("reordered: ", terms)
        #terms = reorder(addtag(terms))
        #print(" ** tagged and reordered:", )
        return terms
    #return node # this is for nodes that aren't Declaration(Bin). not working yet

# keep flatten the way it is. The rewrite will be after, changing [] by Mul() and Add() IR nodes. This process is called
# Future: remove IRconstruct and add Mul() and Add() nodes here already
def flatten(node, op="*"):
    """ separates terms individually (in nodes). At first we separate by `+`, but then all commutative ops """
    print(f"FLATTEN: {node} {op}")
    if isinstance(node, Binary) and op == node.op:
        left = flatten(node.left, op=node.op)
        right = flatten(node.right, op=node.op)
        return left+right
    else:
        return [node]

def addtag(node):
    print("TAG EXPR:\n", node)
    for term in node.operands:
        for i in term.operands:
            if isinstance(i, CudaVar):
                if "thread" in i.base:
                    i.tag = "thread"
                elif "block" in i.base:
                    i.tag = "block"
                else:
                    i.tag = "grid"
            elif isinstance(i, Literal):
                i.tag = "literal"
    return node

# add tag but directly on the node attr instead of creating a tuple. adding one more dim of complexity is bad
#def addtag(terms):
#    """ adds a tag on each node to represent what level that node works with (e.g. thread, block, grid, ...) """
#    print("TAG EXPR:\n", terms)
#    for term in range(len(terms)):
#        if not isinstance(terms[term], list):
#            terms[term] = [terms[term]]
#        for sub in range(len(terms[term])):
#            if isinstance(terms[term][sub], CudaVar):
#                if "thread" in terms[term][sub].base: 
#                    terms[term][sub].tag = "thread"
#                elif "block" in terms[term][sub].base: 
#                    terms[term][sub].tag = "block"
#                else: 
#                    terms[term][sub].tag = "grid"
#            elif isinstance(terms[term][sub], Literal):
#                terms[term][sub].tag = "literal"
#    return terms

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
        mul.operands = sorted(mul.operands, key=lambda x: order.get(x.tag, 99))
        
        print("mul.operands:", mul.operands)
        # fold
        for i in mul.operands:
            print("i: ", i)
            if isinstance(i, Literal):
                #mul.operands = fold(mul, op="*")
                fold(mul, op="*")
                print(f"folded: {mul}\n{mul.operands}")
                break # jumps outside the for i in mul.operands loop

    # outer sort
    node.operands = sorted(node.operands, key=lambda m: order.get(m.operands[0].tag, 99))
    return node


#def reorder(terms):
#    """ reorders the terms based on the tag of each node """
#    print("REORDER ATTR:\n", terms)
#    order = {
#        "thread":  0,
#        "block":   1,
#        "grid":    2,
#        "literal": 3,
#    }
#
#    # (inner sort)
#    for t in range(len(terms)): # terms[t] == term
#        for i in range(len(terms[t])-1): # loop through factors of one term
#            for j in range(1, len(terms[t])):
#                if isinstance(terms[t], list) and len(terms[t]) > 1:
#                    if order.get(terms[t][i].tag) > order.get(terms[t][j].tag):
#                        tmp = terms[t][i]
#                        terms[t][i] = terms[t][j]
#                        terms[t][j] = tmp
#
#        # fold
#        for i in terms[t]:
#            if isinstance(i, Literal):
#                terms[t] = fold(terms[t], op="*") # need to figure it out how to pass the 'op' to 'fold()'
#                #print("returned fold: ", terms[t]) # debug
#                break # added this break to fix terms[t] problem. Not sure if this is right, but it works
#
#    # (outer sort)
#    for t1 in range(len(terms)-1):
#        for t2 in range(1, len(terms)):
#            # it's inner sorted, so first tag will always be the priority tag of that term (sublist)
#            if order.get(terms[t1][0].tag) > order.get(terms[t2][0].tag):
#                tmp = terms[t1]
#                terms[t1] = terms[t2]
#                terms[t2] = tmp
#    #print("outer sorted terms:", terms) # debug
#    return terms

# new fold version
def fold(terms, op="*"):
    print("FOLD: \n", terms)
    assert isinstance(terms, Mul), "Wrong object!"
    acc = 1 if op == "*" else 0
    # keep track of the node types
    literals = [sub for sub in terms.operands if isinstance(sub, Literal)]
    vars = [sub for sub in terms.operands if not isinstance(sub, Literal)]
    print("literals: ", literals)
    print("vars: ", vars)

    for lit in literals:
        print("lit:", lit)
        acc = acc*int(lit.value) if op == "*" else acc+int(lit.value)
        print("acc:", acc)

    if acc == 1:
        terms.operands = vars
    elif acc == 0:
        pass
    else:
        terms.operands = vars + [Literal(value=acc, tag="literal")]

    print("RETURNED: ", terms)
    #return terms

# fixing FOLD for nodes with already Mul() and Add() nodes
#def fold(terms, op="*"):
#    print("fold: ", terms)
#    assert isinstance(terms, Mul), "Wrong object!"
#    folded = terms #copy.copy(terms) #terms.copy()
#    remove = []
#    node = None
#    acc = 0 if op == "+" else 1
#    print("folded: ", folded)
#    print("remove: ", remove)
#    print("node: ", node)
#    print("acc: ", acc)
#
#    for sub in terms.operands:
#        print("sub: ", sub)
#        if isinstance(sub, Literal):
#            if sub.value == "1" and op == "*":
#                print("1*")
#                #folded.remove(sub)
#                #remove.append(sub) # add this?
#            elif sub.value == "0" and op == "*":
#                print("0*")
#                #pass
#            else:
#                print(" ** accumulate **")
#                acc = acc*int(sub.value) if op == "*" else acc+int(sub.value)
#                print(acc)
#                node = sub
#                print(node)
#                remove.append(sub)
#                print(remove)
#        else:
#            continue
#
#    print("folded list:", folded)
#    print("remove list:", remove)
#    print("node: ", node)
#    print("acc: ", acc)
#    if remove != []:
#        node.value = acc
#        print("new node value:", node)
#        for i in remove:
#            if i in folded:
#                folded.remove(i)
#        folded.append(node)
#
#    return folded

# need to rewrite this better! Also, this could be a rewrite rule afterwards.
#def fold(terms, op="*"):
#    """ constant folding to optimize the METAL ast """
#    #print("FOLD_ATTR:\n", terms) # debug
#    assert isinstance(terms, list), "terms to be folded are not list!"
#    folded = terms.copy()
#    remove = []
#    node = None
#    acc = 0 if op == "+" else 1
#
#    for sub in range(len(terms)):
#        if isinstance(terms[sub], Literal):
#            if terms[sub].value == "1" and op == "*":
#                folded.remove(folded[sub])
#            elif terms[sub].value == "0" and op == "*":
#                pass # add this to remove '* 0'
#            else:
#                #print(" ** accumulate **") # debug
#                acc = acc * int(terms[sub].value) if op == "*" else acc + int(terms[sub].value)
#                node = terms[sub]
#                remove.append(terms[sub])
#        else:
#            continue
#    
#    if remove != []:
#        node.value = acc
#        for i in remove:
#            if i in folded:
#                folded.remove(i)
#        folded.append(node)
#    return folded


# adds Mul() and Add() IR nodes. Takes the ordered canonical flattened expr and rewrite with Mul() and Add() nodes.
def IRconstruct(expr):
    print("IR construct:\n", expr)
    for inner in range(len(expr)):
        expr[inner] = Mul(expr[inner])
    newIR = Add(expr)
    print(newIR)
    recognition(newIR)
    print(newIR)
    return newIR
    
# This is IR rewrite. This is the last thing to be called.
def recognition(canonical_expr):
    print("RECOGNITION: \n", canonical_expr)
    node = None
    if canonical_expr is not None:
        for t in canonical_expr.operands:
            for x in range(len(t.operands)):
                if isinstance(t.operands[x], CudaVar): 
                    if t.operands[x].base == "threadIdx": node = ThreadIdx(dim=t.operands[x].dim)
                    elif t.operands[x].base == "blockIdx": node = BlockIdx(dim=t.operands[x].dim)
                    elif t.operands[x].base == "blockDim": node = BlockDim(dim=t.operands[x].dim)
                    t.operands[x] = node


# adding high-level semantic nodes to the expressions
# Add(operands=[Mul(operands=[ThreadIdx(dim='x')]), Mul(operands=[BlockIdx(dim='x'), BlockDim(dim='x')])]) 
# -> GlobalThreadIdx()
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
        "GlobalThreadIdx",  # name
        pat_GlobalThreadIdx,  # pattern function
        build_GlobalThreadIdx # builder function
    )

    rewriter = Rewriter([rule1])
    new_tree = rewriter.rewrite(subtree)
    print("RESULT: ", new_tree)
    return new_tree


# OBS:
# 1- make the IR all in the beginning. So when we flatten the node, already create a high level IR
#    with Mul/Add nodes and ThreadIdx/BlockIdx/BlockDim nodes as well. Make the IR all in the beginning.
#    This is the Lower part, where we change the AST into IR
# 2- canonicalize. So reorder, fold. We can remove addtag, because we can return its value by checking
#    its IR node.
# 3- pattern matching. IR rewrite to higher-level semantic nodes





# TODO:
# right approach:
# 1- create atomic semantic nodes based on the list canonicalized. So ThreadIdx(dim=x), BlockIdx(dim=x), ...
#    replace the nodes on that list for these atomic ones. Even if the sublist means something, only replace to atomic nodes first.
# 2- pattern/rewrite (composition): rule engine to match the atomic nodes configuration into a semantic node:
#    so like: [ThreadIdx(dim=x), [BlockIdx(dim=x), BlockDim(dim=x)]] into -> GlobalThreadIdx(dim=x)
# 3- These composite semantic nodes will be part of the new METAL ast. Then lower the metal ast and codegen
#
# optimizations:
# - improve `fold` function
# - algebraic simplification
# - fuse kernels. Add a 'opt' flag for user if he wants to fuse kernels
# - add tree normalization. ex: Add(a, Add(b, c)) transform into Add(a, b, c)

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