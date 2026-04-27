#from ast_builder import METAL_Kernel, METAL_Parameter, METAL_Body, METAL_Var, METAL_Declaration, METAL_Assignment, METAL_IfStatement, METAL_ForStatement, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Program
#from ast_builder import Parameter, Declaration, Assignment, Binary, Literal, CudaVar, Variable, Array, ThreadIdx, BlockIdx, BlockDim, GlobalThreadIdx, Mul, Add
from ast_builder import *

class CUDAVisitor(object):
    """ Traverse the ast nodes """
    def __init__(self):
        self.kernel_params = []
        self.body = []
        self.thread_idx_dims = {} # for mapping cuda variables dims # maybe is better to have varname: [dims], so for a variable, check all dims that variable has to decide
        self.atomic_bufs = set()
        self.wbuffers = set()
        self.rbuffers = set()
        # for json metadata (if we have multiple kernels we need to reset the list between kernels)
        self.kernel_metadata = {
            "kernel": {
                "kernelName": "",
                "buffers": [],
                "scalars": [],
            }
        }

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
        
        find_atomics(node, self.atomic_bufs) # to change the parameter type (int* -> atomic_int*). So find buffers that are atomic
        print("Atomics: ", self.atomic_bufs)

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
            
            # build metadata to generate dispatcher
            if isinstance(node, Kernel):
                print("node:", node)
                self.kernel_metadata["kernel"]["kernelName"] = node.name
                for i, p in enumerate(node.parameters): 
                    # instead of adding `int*` to type, add `int32` for example, if p is buffer.
                    metadata = {"name": p.name, "type": p.type, "idx": i}
                    if (p.type == "int*" or p.type == "float*"):
                        metadata["type"] = "int" if p.type == "int*" else "float" # int32 figure it out how to work with different sizes
                        if p.name in self.wbuffers and p.name in self.rbuffers:
                            metadata["access"] = "read_write"
                        elif p.name in self.wbuffers:
                            metadata["access"] = "write"
                        else:
                            metadata["access"] = "read"
                        self.kernel_metadata["kernel"]["buffers"].append(metadata)
                    else:
                        self.kernel_metadata["kernel"]["scalars"].append(metadata)
            print("W Buffers:", self.wbuffers)
            print("R Buffers:", self.rbuffers)
            return METAL_Kernel(qualifier, type, name, self.kernel_params, body)

        else:
            print("W Buffers:", self.wbuffers)
            print("R Buffers:", self.rbuffers)
            return METAL_Kernel(qualifier, type, name, [], [])

    def visit_Parameter(self, node, buffer_idx=0):
        # here we need to check if the buffer is an atomic buffer. If it is, we need to generate the node with
        # `atomic_int*` type. But how to do that? we need to somehow check that node, but we check parameter first,
        # then the kernel body, but we need to check first the body, add that to self.atomic_bufs, then change type
        # on Parameter node. Generate the Metal node with the right type. Maybe we can add a pass, where walks the
        # ast and check all atomics, add that node to `atomic_bufs`, and then execute the visit pass normally to
        # generate Metal nodes.
        print("PARAMETER: ", node)
        mem_type = metal_map(node.mem_type)
        node_type = node.type
        if node.type == "int*" or node.type == "float*": #or node.type == "int" or node.type == "float":
            buffer = buffer_idx
            if node.name in self.atomic_bufs:
                node_type = f'atomic_{node.type}'
        elif node.type == "int" or node.type == "float":
            buffer = buffer_idx
            mem_type = "constant"
            node_type = node.type + "&"
        else:
            mem_type = None
            buffer = None
        return METAL_Parameter(memory_type=mem_type, type=node_type, name=node.name, attr=None, buffer=buffer, init=None)

    def visit_Body(self, node):
        if node.children():
            statements = []
            for child in node.children():
                # for each child return the respective METAL node. Ex: visit(child) = METAL_Declaration(type, name, value)
                child_node = self.visit(child) # visit(Declaration), visit(Assignment)
                # check if it's Parameter() node for tid and gid
                if child_node is not None: # this filters in cases like GlobalThreadIdx, because they're added to params, so there's no need to add them in body
                    statements.append(child_node) 
                #elif isinstance(child_node, list):
                #    statements.extend(child_node)
                print("child node:", child_node)
            print("statements", statements)
            return METAL_Body(statements)
        else:
            return METAL_Body(node.statement)

    def visit_Declaration(self, node, parent=None):
        node = pattern_matcher(node) # when we rewrite the IR with GlobalThreadIdx() node for example, the code calls the visit_error() function, because there's no visit_GlobalThreadIdx() node for it. This could be a problem when creating METAL_ast. Fix that later!
        print("DECLARATION:", node)
        memory = node.memory if node.memory else None
        type = node.type
        name = self.visit(node.name) if isnode(node.name) else node.name
        print("mem:", memory)
        print("tye:", type)
        print("name:", name)
        # add a function to remove this r/w buffers appends
        if isinstance(node.name, Array):
            self.wbuffers.append(node.name.name) # change this to call the function `buf_class`

        if isnode(node.value):
            val = self.visit(node.value, parent=node)
            print("val:", val)
            if isinstance(node.value, Binary):
                if isinstance(node.value.left, Array):
                    self.rbuffers.add(node.value.left.name)
                if isinstance(node.value.right, Array):
                    self.rbuffers.add(node.value.right.name)
            elif isinstance(node.value, Array):
                self.rbuffers.add(node.value.name)
        
        if isinstance(node.value, GlobalThreadIdx):
            # we can't use uint3 as array index. Needs to be uint3.x
            param = METAL_Parameter(memory_type=None, type="uint3", name="tid", attr="thread_position_in_grid", buffer=None, init=None)
            self.thread_idx_dims[node.name] = node.value.dim# this is only couting dims if the node is GlobalThreadIdx()
            if not check_param(self.kernel_params, param.attr): # to not add repetitive vars on params
                self.kernel_params.append(param)
            return None
        
        if node.children():
            #value = [] # = Expression(Binary, Literal, Variable, Array)
            # added now NOT WORKING YET!!!!
            #children = node.children()
            print("len::", len(node.children()))
            if len(node.children()) == 1:
                value = self.visit(node.children()[0])
            else:
                value = [self.visit(c) for c in node.children()]
            
            #for child in node.children():
            #    child_node = self.visit(child, parent=node)
            #    value.append(child_node)

            if value != None:
                print("aqui?")
                return METAL_Declaration(metal_map(memory), type, name, value)
            else:
                return METAL_Declaration(metal_map(memory), type, name)
        else:
            return METAL_Declaration(metal_map(memory), type, name, value=node.value)

    # histogram doesn't have assignment statement, that's why `rbuffers/wbuffers` are not being filled
    def visit_Assignment(self, node, parent=None):
        print("ASSIGNMENT: ", node)
        print(node.name)
        print(node.value)
        name = self.visit(node.name, parent=node) if isnode(node.name) else METAL_Variable(node.name)
        print("name:", name)
        if isinstance(node.name, Array):
            self.wbuffers.add(node.name.name) # check if the buffer is to write

        if isnode(node.value):
            val = self.visit(node.value, parent=node)
            print("VAL:", val)
            if isinstance(node.value, Binary):
                if isinstance(node.value.left, Array):
                    self.rbuffers.add(node.value.left.name)
                if isinstance(node.value.right, Array):
                    self.rbuffers.add(node.value.right.name)
            elif isinstance(node.value, Array):
                self.rbuffers.add(node.value.name) 
        elif node.value.isdigit():
            val = METAL_Literal(node.value)
        else:
            val = METAL_Variable(node.value)
        
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
        
        return METAL_ForStatement(init=init, condition=cond, increment=incr, forBody=stmts)

    # still trying to figure this out
    def visit_AtomicOP(self, node, parent=None):
        # important. If func==atomicCAS and value==Literal, we need to create a Declaration node that 
        # assigns that value and pass the address var in `value` field 
        #print("ATOMIC OP:", node)
        func = metal_map(node.func)
        if isinstance(node.addr, Array):
            self.wbuffers.add(node.addr.name) #wbuffer because we're storing in addr the result
        addr = self.visit(node.addr)
        mem_ordering="memory_order_relaxed"
        #value = self.visit(node.value)
        if node.func == "atomicCAS" and isinstance(node.value, Literal):
            #print("entrou aqui")
            decnode = self.visit_Declaration(Declaration(memory="None", type="int", name="atomicValue", value=node.value))
            #print("decnode:", decnode)
            value = 1
            #print("value:", value)
            des = self.visit(node.desired)
            atomic = METAL_AtomicOP(func=func, addr=addr, value=value, desired=des, mem_ordering=mem_ordering)
            return atomic#[decnode, atomic]
        
        else:
            value = self.visit(node.value)
        
        #mem_ordering = "memory_order_relaxed"
        
        #if node.func == "atomicCAS":
           #des = self.visit(node.desired)
           #return METAL_AtomicOP(func=func, addr=addr, value=value, desired=des, mem_ordering=mem_ordering)

        return METAL_AtomicOP(func=func, addr=addr, value=value, mem_ordering=mem_ordering)

    def visit_SyncThreads(self, node, parent=None):
        return METAL_Barrier()

    def visit_FuncCall(self, node, parent=None):
        print("FUNCCALL\n", node)
        name = metal_map(node.name)
        print("name: ", name)
        args = []
        for a in node.args:
            print("a: ", a)
            args.append(self.visit(a))
        print("args: ", args)
        return METAL_FuncCall(name=name, args=args)

    def visit_Binary(self, node, parent=None):
        metal_op = node.op
        if isnode(node.left):
            left = self.visit(node.left, parent=node)
        elif node.left.isdigit():
            left = METAL_Literal(node.left)
        else:
            left = METAL_Variable(node.left)

        if isnode(node.right):
            right = self.visit(node.right, parent=node)
        elif node.right.isdigit():
            right = METAL_Literal(node.right)
        else:
            right = METAL_Variable(node.right)
        
        return  METAL_Binary(metal_op, left, right)

    def visit_Literal(self, node, parent=None):
        return METAL_Literal(node.value)

    def visit_Variable(self, node, parent=None):
        #print("VARIABLE:", node)
        return METAL_Variable(node.name)

    def visit_Array(self, node, parent=None):
        array_name = self.visit(node.name, parent=node) if isnode(node.name) else node.name
        idx = self.visit(node.index) if isnode(node.index) else METAL_Variable(node.index) # need to add elif isdigit(): METAL_Literal()?
        return METAL_Array(array_name, idx)

    def visit_CudaVar(self, node, parent=None):
        metal_var = metal_map(node.base)
        return METAL_Var(metal_var)

    # OBS: Metal doesn't have built-ins so all cudavar must be passed as argument to metal kernel
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
    
    # this is wrong for all ThreadIdx(), BlockIDx(), BlockDim() nodes to add the dims to thread_idx_dims.
    # it was working only for GlobalThreadIdx() nodes, because would add all dims used for that node, and added 
    # them to know how many dims were using in the kernel to calculate the totalSize passed to dispatcher.
    # Don't know how to work with that, needd to check
    def visit_ThreadIdx(self, node, parent=None):
        name = "tid_local"
        if not check_param(self.kernel_params, "thread_position_in_threadgroup"):
            param = METAL_Parameter(memory_type=None, type="uint3", name=name, attr="thread_position_in_threadgroup", buffer=None, init=None)
            self.kernel_params.append(param)
        return METAL_Variable(name=f"{name}.{node.dim}") # it was: name=f"threadIdx.{node.dim}"

    def visit_BlockIdx(self, node, parent=None):
        name = "bid"
        # maybe this is a solution to add all dims used from a variable name "name"
        # But i think only works for blockIdx, threadIdx, blockDim. What about GlobalThreadIdx()? 
        # Maybe the way to analyse is not by length, but to actually checking all dims that have.
        if not check_param(self.kernel_params, "threadgroup_position_in_grid"):
            param = METAL_Parameter(memory_type=None, type="uint3", name=name, attr="threadgroup_position_in_grid", buffer=None, init=None)
            self.kernel_params.append(param)
        return METAL_Variable(name=f"{name}.{node.dim}") # it was: name=f"blockIdx.{node.dim}"
    
    def visit_BlockDim(self, node, parent=None):
        name = "bdim"
        if not check_param(self.kernel_params, "threads_per_threadgroup"):
            param = METAL_Parameter(memory_type=None, type="uint3", name=name, attr=f"threads_per_threadgroup", buffer=None, init=None)
            self.kernel_params.append(param)
        return METAL_Variable(name=f"{name}.{node.dim}") 
    
    # error call function
    def visit_error(self, node, parent=None):
        print("Error: ", node, node.__class__.__name__)
        print(f"No visit method for node: {node.__class__.__name__}")
        print("Node:\n", node)

def check_param(params, attr):
    for p in params:
        if p.attr == attr:
            return True
    return False

# helpers (move this to another file)
def isnode(node):
    """ Check if the node that we're visiting has any node as value for any attribute """
    return isinstance(node, (Binary, Literal, Variable, Array, CudaVar, FuncCall, ThreadIdx, METAL_Binary, METAL_Literal, METAL_Variable, METAL_Array, METAL_Var, METAL_FuncCall))

def buf_class(buf):
    # move all comparision here to append to r/w buffer
    pass

def find_atomics(node, atomic_bufs):
    if isinstance(node, AtomicOP):
        if isinstance(node.addr, Array):
            atomic_bufs.add(node.addr.name) 
    if hasattr(node, 'children'):
        for child in node.children():
            if child is not None:
                find_atomics(child, atomic_bufs)

# i guess i can remove stuff from here, like: "blockIdx.x * blockDim.x + threadIdx.x": metal_term = "[[thread_position_in_grid]]" 
def metal_map(cuda_term):
    """ Maps CUDA concept syntax into METAL concept syntax"""
    metal_term = None
    match cuda_term:
        case "blockIdx":        metal_term = "[[threadgroup_position_in_grid]]"
        case "threadIdx":       metal_term = "[[thread_position_in_threadgroup]]"
        case "blockDim":        metal_term = "[[threads_per_threadgroup]]"
        case "__global__":      metal_term = "device" # using global memory
        case "__shared__":      metal_term = "threadgroup" # using shared memory
        case "__constant__":    metal_term = "constant"
        case "__syncthreads()": metal_term = "threadgroud_barrier()"
        case "atomicAdd":       metal_term = "atomic_fetch_add_explicit"
        case "atomicSub":       metal_term = "atomic_fetch_sub_explicit"
        case "atomicCAS":       metal_term = "atomic_compare_exchange_weak_explicit"
        case "expf":            metal_term = "exp"
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
        print("binds:", binds)
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
    print("Pattern function (GlobalThreadIdx): ", node)
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


# UPDATE:
# shift the focus on writing everything that exists in cuda to metal and focus on real kernels.:
# - softmax
# - attention
# - flash-attention
#
#
# VERY IMPORTANT!!!!!: 
#   1. In metal every kernel parameter must live in a specific address space (device, threadgroup, constant, thread, built-in var)
#      So: `kernel void a(int M)` is wrong! It needs to be: `kernel void a(constant uint& M [[buffer(0)]])`
#      Scalars passed from CPU must live in `constant` address space!
#   2. Why needs `[buffer(0)]` if its scalar?
#      everything coming from CPU is bound into a buffer slot, even if it is just one integer.
#      Metal requires the compiler to know exactly which argument maps to which resource slot in the argument 
#      table. Without [[buffer(index)]], the kernel parameter has no binding location.
#   3. Why `constant uint&` instead of `constant uint`?
#      Because metal passes constant buffer arguments by reference.
#
# TODO:
# - improve `fold` function
# - implement atomicCAS(&address, expected, desired). Change the value on &address to be `desired` and 
# return the old value if the value stored in address is the same as the expected.
# - don't implement native Metal atomics with atomicCAS because its slow. Only implement the ones
# that metal doesn't support like float atomicMax(), atomicMul(). 