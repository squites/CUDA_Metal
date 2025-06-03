from lark import Lark, Transformer, Token, Tree
from dataclasses import dataclass
from typing import Union, List, Optional
from enum import Enum
# grammar
from grammar import cuda_grammar

# classes for the Nodes of the Abstract Syntax Tree. So each class is an Object that will be a node when called.
class ASTNode:
    #pass
    def pretty_print(self, indent=0):
        raise NotImplementedError

# important classes for semantic
@dataclass
class Kernel(ASTNode):
    qualifier: str
    type: str
    name: str
    parameters: List["Parameter"]
    body: "Body"

    def pretty_print(self, indent=0):
        space = " " * (indent+2)
        print("Kernel(")
        print(f"{space}qualifier={self.qualifier},")
        print(f"{space}type={self.type},")
        print(f"{space}name={self.name},")
        print(f"{space}parameters=[")
        for param in self.parameters:
            param.pretty_print(indent+4)
        print(f"{space}],")
        self.body.pretty_print(indent+2)
        print(")") #{space}?

@dataclass
class Parameter(ASTNode):
    type: str
    name: str

    def pretty_print(self, indent):
        space = " " * indent
        print(f"{space}Parameter(type={self.type}, name={self.name})")

@dataclass
class Body(ASTNode):
    statements: List["Statement"]

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Body(statements=[")
        for stmnt in self.statements:
            stmnt.pretty_print(indent+2)
        print(f"{space}])")

# base class for declaration and assignment
class Statement(ASTNode):
    pass

@dataclass
class Declaration(Statement):
    type: str
    name: str
    value: Optional["Expression"] = None

    def pretty_print(self, indent=0):
        space = " " * (indent + 2)
        print("    Declaration(")
        print(f"{space}type={self.type},")
        print(f"{space}name={self.name},")
        print(f"{space}value={self.value}")
        print("    )")

@dataclass
class Assignment(Statement):
    name: str
    value: "Expression" # instance of Expression class. "Expression" with quotes because Expression class is not yet defined

    def pretty_print(self, indent=0):
        space = " " * (indent+2)
        print(f"    Assignment(")
        print(f"{space}name={self.name},")
        print(f"{space}value={self.value}")
        print("    )")

# base class for expressions
class Expression(ASTNode):
    pass

@dataclass
class Binary(Expression):
    op: str
    left: Expression 
    right: Expression

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Binary(")
        print(f"{space}op={self.op},")
        print(f"{space}left={self.left},")
        print(f"{space}right={self.right}")
        print(")")

@dataclass
class Literal(Expression): # constant
    value: Union[int, float]

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Literal(")
        print(f"{space}value={self.value}")
        print(")")

@dataclass
class Variable(Expression): # var name
    name: str

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Variable(")
        print(f"{space}name={self.name}")
        print(")")

# maybe create a new class ArrayAccess
@dataclass
class Array(Expression):
    name: Variable
    index: Expression

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Array(")
        print(f"{space}name={self.name},")
        print(f"{space}index={self.index}")
        print(")")

@dataclass
class CudaVar:
    base: str # blockIdx, threadIdx, ...
    dim: str # x, y, z

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}CudaVar(")
        print(f"{space}base={self.base},")
        print(f"{space}dim={self.dim}")
        print(")")

# Transformer class
class CUDA_AST(Transformer):
    """ 
    Class where you define how each grammar rule (and parse_tree nodes) becomes an AST node. 
    You create a Transformer class with methods for each grammar rule. Each method takes the parse tree's 
    children (tokens or subtrees) and returns an AST node.
    """
    def start(self, items):
        return items[0]
    
    def kernel(self, items):
        qualifier, name, params, body = items
        return Kernel(qualifier=qualifier, type="void", name=str(name), parameters=params, body=body)

    def params(self, items):
        return items

    def parameter(self, items):
        type, name = items
        return Parameter(type=type, name=str(name)) # "'list' object has no attribute 'value'". I took off the ".value"

    def body(self, block):
        return Body(statements=block)

    # Used to replace "Tree(Token('RULE', 'statement'), ...)"
    def statement(self, items):
        return items[0]
    
    def declaration(self, items):
        type, name, initializer = items
        return Declaration(type=str(type), name=str(name), value=initializer)

    def assignment(self, items):
        name, value = items
        return Assignment(name=name, value=value)

    def expression(self, items):
        if len(items) == 1: # single term
            return items[0]
        left = items[0]
        for op, right in zip(items[1::2], items[2::2]):
            left = Binary(op=op, left=left, right=right)
        return left

    def term(self, items):
        if len(items) == 1: # only one factor
            return items[0]
        left = items[0]     # multiple factors
        for op, right in zip(items[1::2], items[2::2]):
            left = Binary(op=op, left=left, right=right)
        return left

    def factor(self, items):
        item = items[0]
        if isinstance(item, Token): 
            if item.type == "NUMBER":
                return Literal(value=item.value)
            elif item.type == "NAME":
                return Variable(name=item.value)
        return item#.value

    def qualifier(self, token):
        return token[0].value

    def type(self, token):
        return token[0].value   

    def term_ops(self, token):
        return token[0].value

    def factor_ops(self, token):
        return token[0].value
    
    def identifier(self, token):
        return token[0] #.value

    def array_index(self, items):
        name, index = items
        return Array(name=str(name), index=index)

    def base_var(self, items):
        #print(f"base_var: {items}")
        return str(items[0].value)

    def cuda_dim(self, items):
        #print(f"cuda_dim: {items}")
        return str(items[0].value)
        
    def cuda_var(self, items):
        #print(f"cuda_var items: {items}")
        base, dim = items
        return CudaVar(base=str(base), dim=str(dim))


# after move this kernel to "kernel.cu"
kernel = r"""
__global__ void add(int a, int b) {
    int c = a + b;
}
"""

kernel_vecAdd = r"""
__global__ void vecAdd(int* a, int* b, int* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];  
}
"""

parser = Lark(cuda_grammar)
tree = parser.parse(kernel_vecAdd)
print(f"parse tree: {tree}")
#print(tree.pretty())

x = CUDA_AST()
ast = x.transform(tree)
#print(ast.pretty())

#ast.pretty_print() # structured print

# "binary" = 2 operands and 1 operator: 2 + 3 or 2 + (3 * 4) or a + b
# "literal" = constant value: 2 or 3.1 or "string"
# "variable" = a named identifier: a or result or threadIdx

# Expression is "Term((+|-) Term)*" . Ex: (8 + 24 * 2) is an expression because there are one Term '8' + another Term '24 * 2'
# Term is "Factor((*|/) Factor)*" Ex: (8) + (24 * 2), so 8 is a term and (24 * 2) is another term, because they are splitted by + signal
# Factor are the numbers. Ex: 8 + 24 * 2, so 8, 24, 2 are all factors


# Transformer funcionality:
# In parse tree only has Tree() or Token(). Tree(data=rule_name, children=[...]) while Token(type=TERMINAL_SYMBOL, value). 
# Tree() represent grammar rules. Token() represent terminal symbols (leaves).
#
# Imagine we have:
# Tree(Token('RULE', 'qualifier'), [Token('QUALIFIER', '__global__')])
# 1) - data field is a Token('RULE', 'qualifier'), which means that the rule_name is 'qualifier'
#    - has one child [Token('QUALIFIER', '__global__')], which is a terminal token with the string '__global__'
#
# 2) Transformer sees the Tree(...)
#    - extracts the rule_name from data which is "qualifier"
#    - it looks for a method in Transformer class called "def qualifier()"
#
# 3) Before calling the method, the Transformer recursively transforms the children
#    - the child is a Token(), so converts to its string value which is "__global__"
#    - after doing this with all the children, it calls the qualifier method passing the children as parameters
#      so: qualifier(self, ["__global__"])
# 
# 4) The qualifier method receives ["__global__"]:
#    - then it can create an AST node calling the corresponded class "Qualifier(...)"  or simply returns the string.
#
# When to return a string and when to create a AST node?
# Node: when represents a semantic structure. is not just a string, but a construct with meaning and behavior
# string: when is just a primitive value like: "int", 'x', "__global__", ...