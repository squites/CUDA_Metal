#from main import METAL_Ast
from traverse import isnode
from ast_builder import METAL_Ast, METAL_Kernel, METAL_Parameter, METAL_IfStatement, METAL_ForStatement

class CodeGen():
    """
    Walk METAL ast and generate a string which is the metal code 
    """
    #def __init__(self, metal_str):
    #    self.metal_ast = metal_ast
    #    self.metal_str = metal_str # code string

    def generator(self, node, indent=0): # add indent. For specific calls like inside for params, we can set right indent
        """ 
        Args:
            node: METAL ast node to generate that specific string
        """
        #print(f"node: {node}") #debug
        method = "gen_" + node.__class__.__name__
        gen = getattr(self, method, self.gen_error)
        #print(f"method: {method}") #debug
        return gen(node, indent)

    def gen_error(self):
        print("Error!")

    def gen_METAL_Program(self, node, indent=0):
        if node.header == None:
            header = "#include <metal_stdlib>"
            namespace = "using namespace metal"
            #kernel = self.generator(node.kernel)
            kernel = self.generator(node.kernel)
        return f"{header}\n{namespace};\n{kernel}"

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
    #def gen_METAL_Kernel(self, node, indent=0):
    #    qualifier = node.qualifier
    #    type = node.type
    #    name = node.name
    #    metal_code = f"{qualifier} {type} {name}("
    #    if node.children():
    #        i = 1
    #        for child in node.children():
    #            if isinstance(child, METAL_Parameter):
    #                if len(node.parameters) == i:
    #                    child_node_str = self.generator(child, indent=0)
    #                    bodystr = child_node_str + ") {\n"
    #                else:
    #                    child_node_str = self.generator(child, indent=9)
    #                    bodystr = child_node_str
    #                i += 1
    #            else:
    #                child_node_str = self.generator(child, indent=4)
    #            metal_code = metal_code + child_node_str
    #    self.metal_str = metal_code + "\n}"
    #    return self.metal_str

    def gen_METAL_Parameter(self, node, indent=9):
        space = " " * indent
        memory_type = node.memory_type
        type = node.type
        name = node.name
        buff = "" if not node.buffer else str(node.buffer)
        code_str = f"{str(memory_type)} {str(type)} {str(name)} {str(buff)}"
        return code_str 

    def gen_METAL_Body(self, node, indent=0):
        body_str = ""
        if node.children():
            for child in node.children():
                child_body_str = self.generator(child, indent=4) # body statment must have 4 tabs
                body_str = body_str + child_body_str
        return body_str

    def gen_METAL_Declaration(self, node, indent):
        print(node.name)
        space = " " * indent
        type = node.type
        name = node.name
        declaration_str = f"{type} {name} = "
        if node.children():
            for child in node.children():
                #print(f"decla child: {child}")
                child_decla_str = self.generator(child, indent)
                #print(f"child_decla_str: {child_decla_str}")
                declaration_str = declaration_str + child_decla_str
        return space + declaration_str + ";" #\n this \n is what makes 

    def gen_METAL_Assignment(self, node, indent):
        space = " " * indent
        name = self.generator(node.name) if isnode(node.name) else str(node.name)
        val = self.generator(node.value) if isnode(node.value) else str(node.value)
        assignment_str = f"{name} = {val}"
        return space + assignment_str

    def gen_METAL_IfStatement(self, node, indent):
        space = " " * indent
        cond = self.generator(node.condition) # idx < n
        ifstr = space + "if (" + cond + ") {\n"
        bodystr = f"{space}"
        for b in node.if_body:
            body = self.generator(b, indent=8) # if body statement must have 8 tabs
            bodystr = bodystr + str(body) + ";\n"
        return ifstr + bodystr + space + "}"

    def gen_METAL_ForStatement(self, node, indent):
        space = " " * indent
        init = self.generator(node.init, indent=0)
        cond = self.generator(node.condition, indent=0)
        incr = self.generator(node.increment, indent=0)
        header = space + "for(" + init + cond + ";" + incr + ") {\n"
        bodystr = f"{space}"
        for statement in node.forBody:
            body = self.generator(statement, indent=4)
            bodystr = bodystr + str(body) + ";\n"

        return header + bodystr + space + "}"


    # ex: METAL_Binary(op='+',left=METAL_Variable(name='a'), right='b')
    def gen_METAL_Binary(self, node, indent=0): 
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

    def gen_METAL_Literal(self, node, indent=0):
        return node.value

    def gen_METAL_Variable(self, node, indent=0):
        return str(node.name) 

    def gen_METAL_Array(self, node, indent=0):
        name = self.generator(node.name) if isnode(node.name) else node.name
        idx = self.generator(node.index) if isnode(node.index) else node.index
        array_str = f"{name}[{idx}]"
        return array_str

    def gen_METAL_Var(self, node, indent=0):
        metal_var_str = f"{node.metal_var}"
        return metal_var_str




# Trying to linearize the code first and then indent
#class CodeGen():
#    """
#    Walk METAL ast and generate a linearized string which is the metal code 
#    """
#    def __init__(self):
#        self.metal_str = "" # code string
#
#    def generator(self, node, indent=0): # add indent. For specific calls like inside for params, we can set right indent
#        """ 
#        Args:
#            node: METAL ast node to generate that specific string
#        """
#        method = "gen_" + node.__class__.__name__
#        gen = getattr(self, method, self.gen_error)
#        return gen(node, indent)
#
#    def gen_error(self):
#        print("Error!")
#
#    def gen_METAL_Program(self, node, indent=0):
#        if node.header == None:
#            header = "#include <metal_stdlib>"
#            namespace = "using namespace metal"
#            kernel = self.generator(node.kernel)
#            self.metal_str += f"{header} {namespace}; {kernel}"
#            print(f"\nMETAL_STR: {self.metal_str}")
#        return f"{header} {namespace}; {kernel}"
#        #return self.metal_str
#
#    def gen_METAL_Kernel(self, node, tab=2):
#        qualifier = node.qualifier
#        type = node.type
#        name = node.name
#        metal_code = f"{qualifier} {type} {name}(" # "kernel void vecAdd("
#        if node.children():
#            i = 1
#            for child in node.children():
#                child_node_str = self.generator(child) # return the str generated on each gen method
#                if isinstance(child, METAL_Parameter):
#                    if len(node.parameters) == i:
#                        s = child_node_str + ") {" #+ f"{str(indent)}"
#                    else: 
#                        s = child_node_str + ", " # melhorar esse indent e deixá-lo dinamico
#                    i += 1
#                else:
#                    s = child_node_str + ";"
#                metal_code = metal_code + s
#        #self.metal_str += metal_code + "}"
#        return metal_code + "}"
#
#    def gen_METAL_Parameter(self, node, indent=9):
#        memory_type = node.memory_type
#        type = node.type
#        name = node.name
#        buff = "" if not node.buffer else str(node.buffer)
#        code_str = f"{str(memory_type)} {str(type)} {str(name)} {str(buff)}"
#        return code_str 
#
#    def gen_METAL_Body(self, node, indent=0):
#        body_str = ""
#        if node.children():
#            for child in node.children():
#                child_body_str = self.generator(child, indent=4) # body statment must have 4 tabs
#                body_str = (body_str + child_body_str) if isinstance(child, (METAL_IfStatement, METAL_ForStatement)) else (body_str + child_body_str + "; ")
#        return body_str
#
#    def gen_METAL_Declaration(self, node, indent):
#        type = node.type
#        name = node.name
#        declaration_str = f"{type} {name} = "
#        if node.children():
#            for child in node.children():
#                child_decla_str = self.generator(child, indent)
#                declaration_str = declaration_str + child_decla_str
#        return declaration_str #+ "; " #\n this \n is what makes 
#
#    def gen_METAL_Assignment(self, node, indent):
#        name = self.generator(node.name) if isnode(node.name) else str(node.name)
#        val = self.generator(node.value) if isnode(node.value) else str(node.value)
#        assignment_str = f"{name} = {val}"
#        return assignment_str
#
#    def gen_METAL_IfStatement(self, node, indent):
#        cond = self.generator(node.condition) # idx < n
#        ifstr = "if (" + cond + ") {"
#        bodystr = "" #f"{space}"
#        for b in node.if_body:
#            body = self.generator(b, indent=8) # if body statement must have 8 tabs
#            bodystr += self.generator(b) + "; "
#        return ifstr + bodystr + "}"
#
#    def gen_METAL_ForStatement(self, node, indent):
#        init = self.generator(node.init, indent=0)
#        cond = self.generator(node.condition, indent=0)
#        incr = self.generator(node.increment, indent=0)
#        #if isinstance(node.parent, Statement)
#        header = "for(" + init + "; " + cond + "; " + incr + ") {"
#        bodystr = "" #f"{space}"
#        for statement in node.forBody:
#            body = self.generator(statement, indent=4)
#            if isnode(statement):
#                bodystr = bodystr + str(body) + ";"
#            else:
#                bodystr = bodystr + str(body)
#
#        return header + bodystr + "}"
#
#
#    # ex: METAL_Binary(op='+',left=METAL_Variable(name='a'), right='b')
#    def gen_METAL_Binary(self, node, indent=0): 
#        op = node.op
#        binary_str = ""
#        left = self.generator(node.left) if isnode(node.left) else str(node.left)
#        right = self.generator(node.right) if isnode(node.right) else node.right
#        binary_str = f"{left} {op} {right}"
#        # need to put this in parameter later
#        if binary_str == "[[threadgroup_position_in_grid]] * [[threads_per_threadgroup]] + [[thread_position_in_threadgroup]]":
#            binary_str = "[[thread_position_in_grid]]" # gambiarra (fix later!!!!)
#        return binary_str #+ "; "
#
#    def gen_METAL_Literal(self, node, indent=0):
#        return node.value
#
#    def gen_METAL_Variable(self, node, indent=0):
#        return str(node.name) 
#
#    def gen_METAL_Array(self, node, indent=0):
#        name = self.generator(node.name) if isnode(node.name) else node.name
#        idx = self.generator(node.index) if isnode(node.index) else node.index
#        array_str = f"{name}[{idx}]"
#        return array_str
#
#    def gen_METAL_Var(self, node, indent=0):
#        metal_var_str = f"{node.metal_var}"
#        return metal_var_str

    #def indent(self, metalstr):
    #    indent = 0
    #    space = " " * indent
    #    newstr = ""
    #    for c in metalstr:
    #        if c == '{':
    #            indent += 4
    #            newstr += (space * indent) + str(c)
    #        else:
    #            newstr += str(c)
    #    print(newstr)


