# TODO: implement my own transformer class to visit the parse tree and convert them into AST nodes
#from ast_builder import CUDA_Ast, Declaration, Assignment, CudaVar, Array, Binary, Variable, Literal
from ast_builder import METAL_Kernel_node, METAL_Parameter_node, METAL_Body_node, METAL_Var_node, METAL_Declaration_node, METAL_Assignment_node, METAL_Expression_node, METAL_Binary_node, METAL_Literal_node, METAL_Variable_node, METAL_Array_node

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

    def visit_Kernel(self, node):
        print(f"Kernel node: {node}")
        if node.children():
            for child in node.children():
                print(f"Kernel child node: {child}")
                self.visit(child)
            attrs = []
            for val in node.__dict__.values(): # print only the dict values
                if str(val) == "__global__":
                    val = "kernel"
                attrs.append(val)
            # generate corresponded METAL node
            metal_node = METAL_Kernel_node(attrs)
        else:
            print(f"Node {node} has no children!")
            metal_node = None
        return metal_node
        
    def visit_Parameter(self, node):
        print(f"Parameter node: {node}")
        #self.visit(node) # no need for this call. self.visit() is only needed when the node has children 
        # metal
        return METAL_Parameter_node(memory_type="device", type=node.type, name=node.name)

    def visit_Body(self, node):
        print(f"Body node: {node}")
        if node.children():
            for child in node.children():
                print(f"Body child node: {child}")
                self.visit(child)
            statements = node.statements
            metal_node = METAL_Body_node(statements)
        else:
            print(f"The node {node} has no children!")
            metal_node = None
        return metal_node

    def visit_Statement(self, node):
        pass

    def visit_Declaration(self, node):
        print(f"Declaration node: {node}")
        if node.children():
            for child in node.children():
                print(f"Declaration child node: {child}")
                self.visit(child)
            type = node.type
            name = node.name
            #value = METAL_Expression_node() # wrong
            # value IS WRONG!
            val = node.value
            if "Binary" in str(val):
                metal_op = getattr(val, op)
                metal_l = getattr(val, left)
                metal_r = getattr(val, right)
                metal_node = METAL_Binary_node(metal_op, metal_l, metal_r)
            elif "Literal" in str(node.value):
                metal_val = getattr(val, value)
                metal_node = METAL_Literal_node(metal_val)
            else:
                metal_name = getattr(val, name)
                metal_node = METAL_Variable_node(metal_name)
            return METAL_Declaration_node(type, name, metal_node)
        
        else:
            print(f"Node {node} has no children.")
            return None

    def visit_Assignment(self, node):
        print(f"Assigment node: {node}")
        #self.visit(node) # do we need this function even when there's no children?
        name = node.name
        val = node.value #WRONG! value: Binary(op='+', left=...) must be METAL_Binary_node(op='+', ...)
        if "Binary" in str(value):
            metal_op = getattr(val, op)
            metal_l = getattr(val, left)
            metal_r = getattr(val, right)
            metal_node = METAL_Binary_node(metal_op, metal_l, metal_r)
        elif "Literal" in str(value):
            metal_val = getattr(val, value)
            metal_node = METAL_Literal_node(metal_val)
        else:
            metal_name = getattr(val, name)
            metal_node = METAL_Variable_node(metal_name)
        return METAL_Assignment_node(name, metal_node)

    def visit_Expression(self, node):
        pass

    def getexpr(expr)
        pass # returns the node (Binary, Literal, Variable)

    def visit_Binary(self, node):
        print(f"Binary node: {node}")
        #self.visit(node)
        oper = node.op
        left = METAL_Expression_node() # wrong
        right = METAL_Expression_node() # wrong
        # add these for both (left and right):
        if "Binary" in str(left):
            metal_op = getattr(node.value, op)
            metal_l = getattr(node.value, left)
            metal_r = getattr(node.value, right)
            metal_node = METAL_Binary_node(metal_op, metal_l, metal_r)
        elif "Literal" in str(left):
            metal_val = getattr(node.value, value)
            metal_node = METAL_Literal_node(metal_val)
        else:
            metal_name = getattr(node.value, name)
            metal_node = METAL_Variable_node(metal_name)
        
        return = METAL_Binary_node(oper, metal_node, metal_node)


    def visit_Literal(self, node):
        print(f"Literal node: {node}")
        self.visit(node)
        type = node.type
        metal_node = METAL_Literal_node(type)
        return metal_node

    def visit_Variable(self, node):
        print(f"Variable node: {node}")
        self.visit(node)
        name = node.name
        metal_node = METAL_Variable_node(name)
        return metal_node

    def visit_Array(self, node):
        print(f"Array node: {node}")
        self.visit(node)
        name = node.name
        index = node.index
        metal_node = METAL_Array_node(name, index)
        return metal_node

    def visit_CudaVar(self, node):
        if node.base == "blockIdx":
            metal_var = "[[threadgroup_position_in_grid]]"
        elif node.base == "threadIdx":
            metal_var = "[[thread_position_in_threadgroup"
        elif node.base == "blockDim":
            metal_var = "[[threads_per_threadgroup]]"
        else:
            metal_var = ""
        
        metal_node = METAL_Var_node(metal_var)
        return metal_node

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
