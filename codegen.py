#from main import METAL_Ast
from traverse import isnode
from ast_builder import METAL_Ast, METAL_Kernel, METAL_Parameter

class CodeGen():
    """
    Walk METAL ast and generate a string which is the metal code 
    """
    #def __init__(self, metal_str):
    #    self.metal_ast = metal_ast
    #    self.metal_str = metal_str # code string

    def generator(self, node):
        """ 
        Args:
            node: METAL ast node to generate that specific string
        """
        #print(f"node: {node}") #debug
        method = "gen_" + node.__class__.__name__
        gen = getattr(self, method, self.gen_error)
        #print(f"method: {method}") #debug
        return gen(node)

    def gen_error(self):
        print("Error!")

    def gen_METAL_Kernel(self, node, tab=2):
        indent = " " * (tab)
        qualifier = node.qualifier
        type = node.type
        name = node.name
        metal_code = f"{qualifier} {type} {name}(" # "kernel void vecAdd("
        if node.children():
            i = 1
            for child in node.children():
                #print(f"child: {child}")
                child_node_str = self.generator(child) # return the str generated on each gen method
                #print(f"child_node_str: {child_node_str}\n")
                if isinstance(child, METAL_Parameter):
                    if len(node.parameters) == i:
                        #print(f"i: {i}") # debug
                        s = child_node_str + ") {\n" #+ f"{str(indent)}"
                    else: 
                        s = child_node_str + f",\n{str(indent*9)} " # melhorar esse indent e deixá-lo dinamico
                    #print(f"s: {s}") #debug
                    i += 1
                else:
                    s = child_node_str + ";"
                metal_code = metal_code + s 
        self.metal_str = metal_code + "\n}"
        return self.metal_str

    def gen_METAL_Parameter(self, node):
        memory_type = node.memory_type
        type = node.type
        name = node.name
        code_str = f"{str(memory_type)} {str(type)} {str(name)} {str(node.buffer)}"
        return code_str 

    def gen_METAL_Body(self, node):
        body_str = ""
        #print(f"BODY node: {node}\n") # debug
        if node.children():
            for child in node.children():
                #print(f"body child: {child}") # debug
                child_body_str = self.generator(child)
                #print(f"body child_node_str: {child_body_str}") #debug
                body_str = body_str + child_body_str
        return body_str

    def gen_METAL_Declaration(self, node):
        type = node.type
        name = node.name
        declaration_str = f"{type} {name} = "
        if node.children():
            for child in node.children():
                #print(f"decla child: {child}")
                child_decla_str = self.generator(child)
                #print(f"child_decla_str: {child_decla_str}")
                declaration_str = declaration_str + child_decla_str
        return "    " + declaration_str + ";\n"

    def gen_METAL_Assignment(self, node):
        name = self.generator(node.name) if isnode(node.name) else str(node.name)
        val = self.generator(node.value) if isnode(node.value) else str(node.value)
        assignment_str = f"{name} = {val}"
        return "    " + assignment_str

    # ex: METAL_Binary(op='+',left=METAL_Variable(name='a'), right='b')
    def gen_METAL_Binary(self, node): 
        op = node.op
        binary_str = ""
        left = self.generator(node.left) if isnode(node.left) else str(node.left)
        right = self.generator(node.right) if isnode(node.right) else node.right
        binary_str = f"{left} {op} {right}"
        # need to put this in parameter later
        if binary_str == "[[threadgroup_position_in_grid]] * [[threads_per_threadgroup]] + [[thread_position_in_threadgroup]]":
            binary_str = "[[thread_position_in_grid]]" # gambiarra (fix later!!!!)
        #print(f"binary_str: {binary_str}")
        return binary_str

    def gen_METAL_Literal(self, node):
        return "literal"

    def gen_METAL_Variable(self, node):
        return str(node.name) 

    def gen_METAL_Array(self, node):
        name = self.generator(node.name) if isnode(node.name) else node.name
        idx = self.generator(node.index) if isnode(node.index) else node.index
        array_str = f"{name}[{idx}]"
        return array_str

    def gen_METAL_Var(self, node):
        #print(f"metal var: {node.metal_var}") # debug
        metal_var_str = f"{node.metal_var}"
        return metal_var_str
