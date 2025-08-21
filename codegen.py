#from main import METAL_Ast
from traverse import isnode
from ast_builder import METAL_Parameter, METAL_IfStatement, METAL_ForStatement

class CodeGen():

    def generator(self, node, indent=0): # add indent. For specific calls like inside for params, we can set right indent
        """ 
        Visit the node and generate the code string of that node
        Args:
            node: METAL ast node to generate that specific string
        """
        method = "gen_" + node.__class__.__name__
        gen = getattr(self, method, self.gen_error)
        return gen(node, indent)

    def gen_error(self):
        print("Error!")

    def gen_METAL_Program(self, node, indent=0):
        if node.header == None:
            header = "#include <metal_stdlib>"
            namespace = "using namespace metal"
            kernel = self.generator(node.kernel)
        return f"{header}\n{namespace};\n{kernel}"

    def gen_METAL_Kernel(self, node, indent=0):
        qualifier = node.qualifier
        type = node.type
        name = node.name
        metal_code = f"{qualifier} {type} {name}(" # "kernel void vecAdd("
        if node.children():
            i = 1
            for child in node.children():
                child_node_str = self.generator(child) # return the str generated on each gen method
                print("METAL CHILD: ", child)
                print("METAL_child_node_str: ", child_node_str)
                if isinstance(child, METAL_Parameter):
                    if len(node.parameters) == i:
                        s = child_node_str + ") {\n" #+ f"{str(indent)}"
                    else:
                        space = " " * 25 
                        s = child_node_str + f",\n{space}" # melhorar esse indent e deixá-lo dinamico
                    i += 1
                else:
                    s = child_node_str #+ ";"
                metal_code = metal_code + s
        self.metal_str = metal_code + "\n}"
        return self.metal_str

    def gen_METAL_Parameter(self, node, indent=9):
        space = " " * indent
        memory_type = node.memory_type
        type = node.type
        name = node.name
        buff = "" if not node.buffer else str(node.buffer)
        code_str = f"{str(memory_type)} {str(type)} {str(name)} {str(buff)}"
        return code_str 

    def gen_METAL_Body(self, node, indent=0):
        bodystr = ""
        if node.children():
            for child in node.children():
                print("body child: ", child)
                child_body_str = self.generator(child, indent=4) # body statment must have 4 tabs
                if isinstance(child, (METAL_IfStatement, METAL_ForStatement)):
                    bodystr = bodystr + child_body_str + "\n" # if add ";" the ifstatement will also have ";" which is wrong
                else:
                    bodystr = bodystr + child_body_str + ";\n"
        return bodystr

    def gen_METAL_Declaration(self, node, indent):
        space = " " * indent # indent=4 for body child nodes
        type = node.type
        name = node.name
        declaration_str = f"{type} {name} = "
        if node.children():
            for child in node.children():
                child_decla_str = self.generator(child)#, indent)
                declaration_str = declaration_str + child_decla_str
        return space + declaration_str# + ";" 

    def gen_METAL_Assignment(self, node, indent):
        space = " " * indent
        name = self.generator(node.name) if isnode(node.name) else str(node.name)
        val = self.generator(node.value) if isnode(node.value) else str(node.value)
        assignment_str = f"{name} = {val}"
        return space + assignment_str

    def gen_METAL_IfStatement(self, node, indent):
        space = " " * indent # indent of "if" inside kernel body is 4
        cond = self.generator(node.condition)
        ifstr = space + "if (" + cond + ") {\n"
        bodystr = ""
        for b in node.if_body:
            print("body: ", b)
            body = self.generator(b, indent=indent+4)
            if isinstance(b, (METAL_ForStatement, METAL_IfStatement)):
                bodystr = bodystr + str(body) + "\n"
            else:
                bodystr = bodystr + str(body) + ";\n"
        return ifstr + bodystr + space + "}"

    def gen_METAL_ForStatement(self, node, indent):
        space = " " * indent
        init = self.generator(node.init)
        cond = self.generator(node.condition)
        incr = self.generator(node.increment)
        header = space + "for (" + init + " " + cond + "; " + incr + ") {\n"
        bodystr = ""
        for statement in node.forBody:
            body = self.generator(statement, indent=indent+4)
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