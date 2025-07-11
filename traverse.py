# TODO: implement my own transformer class to visit the parse tree and convert them into AST nodes
#from ast_builder import CUDA_Ast, Declaration, Assignment, CudaVar, Array, Binary, Variable, Literal
from ast_builder import METAL_Kernel_node, METAL_Parameter_node, METAL_Body_node, METAL_Var_node, METAL_Declaration_node, METAL_Assignment_node, METAL_Expression_node, METAL_Binary_node, METAL_Literal_node, METAL_Variable_node, METAL_Array_node
from ast_builder import Parameter, Binary, Literal, Variable
# Visitor
class CUDAVisitor(object): # CUDATraverse()
    """ Traverse the ast nodes """
    def visit(self, node): # 1st call node: <class '__main__.Kernel'>
        #print(f"Visiting node: {type(node)}")
        method = "visit_" + node.__class__.__name__ # attribute
        visitor = getattr(self, method, self.visit_error) # getattr(obj, attr, default): returns the value of the attribute "attr" of the object "obj". returns a reference of the function "method"
        # visitor = getattr(self, "visit_Kernel")
        # visitor() is the same as self.visit_kernel() for example
        # visitor(node): we pass the root node "Kernel()"" on 1st call.
        #print(f"visitor: {visitor} visitor method: {visit_method}")
        return visitor(node) # visit_kernel(ast_builder.Kernel)

    # OK
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
                    params.append(child_node) # append(METAL_Parameter_node)
                else:
                    body.append(child_node) # append(METAL_Body_node)

            # generate corresponded METAL node
            return METAL_Kernel_node(qualifier, type, name, params, body)
        else:
            print(f"Node {node} has no children!")
            return METAL_Kernel_node(qualifier, type, name, [], [])

    # OK   
    def visit_Parameter(self, node):
        print(f"Parameter node: {node}")
        #self.visit(node) # no need for this call. self.visit() is only needed when the node has children 
        return METAL_Parameter_node(memory_type="device", type=node.type, name=node.name)

    # OK
    def visit_Body(self, node):
        print(f"Body node: {node}") # debug
        if node.children():
            statements = []
            for child in node.children():
                print(f"Body child node: {child}")
                # for each child, will return the respective METAL node. Ex: visit(child) = METAL_Declaration_node(type, name, value)
                child_node = self.visit(child) # visit(Declaration), visit(Assignment)
                statements.append(child_node) # this right? It is if child_node is being a METAL node returned by visit methods. Not right if child_node is CUDA node
            #statements = [METAL_Declaration_node(...), METAL_Assignment_node(...)]
            return METAL_Body_node(statements)
        else:
            print(f"The node {node} has no children!")
            return METAL_Body_node(node.statement)

    # OK
    def visit_Statement(self, node):
        pass

    # TODO OK?
    def visit_Declaration(self, node): #visit_Declaration(Declaration())
        print(f"Declaration node: {node}")
        type = node.type
        name = node.name
        
        if node.children():
            value = [] # It'll be an Expression(Binary, Literal, Variable, Array)
            for child in node.children():
                print(f"Declaration child node: {child}")
                child_node = self.visit(child) # this is equal as: "return METAL_Declaration_node(type, name, value)"
                #expr = get_expr(child_node) # check if its Binary, Literal, Variable
                value.append(child_node)
            return METAL_Declaration_node(type, name, value)
        
        else:
            print(f"Node {node} has no children.")
            return METAL_Declaration_node(type, name, None)

    # OK
    def visit_Assignment(self, node):
        print(f"Assigment node: {node}")
        name = node.name
        val = node.value #value: Binary(op='+', left=...) must be METAL_Binary_node(op='+', ...)
        if isinstance(val, Binary): # if val is Binary node
            metal_op = getattr(val, "op")
            metal_l = getattr(val, "left")
            metal_r = getattr(val, "right")
            metal_node = METAL_Binary_node(metal_op, metal_l, metal_r)
        elif isinstance(val, Literal):
            metal_val = getattr(val, "value")
            metal_node = METAL_Literal_node(metal_val)
        else:
            metal_name = getattr(val, "name")
            metal_node = METAL_Variable_node(metal_name)

        return METAL_Assignment_node(name, metal_node) # ex: METAL_Assignment_node("x", METAL_Binary_node())

    # OK
    def visit_Expression(self, node):
        pass

    # OK
    def get_expr(self, node):
        #pass # returns the node (Binary, Literal, Variable)
        if isinstance(node, Binary):
            metal_op = getattr(node, "op")
            metal_l = getattr(node, "left")
            metal_r = getattr(node, "right")
            metal_node = METAL_Binary_node(metal_op, metal_l, metal_r)
        elif isinstance(node, Literal):
            metal_val = getattr(node, "value")
            metal_node = METAL_Literal_node(metal_val)
        else:
            metal_name = getattr(node, "name")
            metal_node = METAL_Variable_node(metal_name)
        # Should I add METAL_Array_node()? Because inherits from METAL_Expression_node

        return metal_node

    # OK
    def visit_Binary(self, node):
        print(f"Binary node: {node}")
        metal_op = node.op
        left = self.get_expr(node.left)
        right = self.get_expr(node.right)
        return  METAL_Binary_node(metal_op, left, right)

    # OK
    def visit_Literal(self, node):
        print(f"Literal node: {node}")
        type = node.type
        return METAL_Literal_node(type)

    # OK
    def visit_Variable(self, node):
        print(f"Variable node: {node}")
        name = node.name
        return METAL_Variable_node(name)

    # OK
    def visit_Array(self, node):
        print(f"Array node: {node}")
        array_name = getattr(node.name, "name") # Variable(name)
        name = METAL_Variable_node(array_name)
        idx = get_expr(node.index)
        return METAL_Array_node(name, idx)

    # OK
    def visit_CudaVar(self, node):
        if node.base == "blockIdx":
            metal_var = "[[threadgroup_position_in_grid]]"
        elif node.base == "threadIdx":
            metal_var = "[[thread_position_in_threadgroup"
        elif node.base == "blockDim":
            metal_var = "[[threads_per_threadgroup]]"
        else:
            metal_var = ""

        return METAL_Var_node(metal_var)

    def visit_error(self, node, attr):
        print(f"The node {node} has no attribute named {attr}!") 

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
