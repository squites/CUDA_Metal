# TODO: implement my own transformer class to visit the parse tree and convert them into AST nodes
from ast_builder import cuda_ast, ASTNode, Declaration, Assignment


class ASTVisitor(ASTNode):
    #def __init__(self, cuda_ast):
    #    self.cuda_ast = cuda_ast

    def visit(self, cuda_ast):
        # if node exists, do something
        #if str(node) == "kernel":
        self.visit_kernel(cuda_ast)
        #elif str(node) == "parameter":
        #    visit_parameter(self)
        #elif str(node) == "body":
        #    visit_body(self.cuda_ast)
        #else:
        #    visit_statement(self.cuda_ast)
        pass

    def visit_kernel(self, ast):
        print(f"kernel: {ast}\n")
        self.visit_body(ast.body)
        for param in ast.parameters:
            self.visit_parameter(param)

    def visit_body(self, node):
        print(f"body: {node}\n")
        for stmt in node.statements:
            self.visit_statement(stmt)

    def visit_parameter(self, node):
        print(f"parameter: {node}")
        pass

    def visit_statement(self, node):
        #if isinstance(s, Declaration):
        #    visit_declaration()
        #elif isinstance(s, Assignment):
        #    visit_assignment()
        #else:
        #    visit_expression()
        pass

visitor = ASTVisitor()
visitor.visit(cuda_ast)

# add metal mapping for each visit() function

    