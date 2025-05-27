from lark import Lark, Transformer, Token
from dataclasses import dataclass
from typing import Union, List, Optional
from enum import Enum

# grammar
from grammar import cuda_grammar

# class for the Node of the Abstract Syntax Tree
class ASTNode:
    pass

# important classes for semantic
@dataclass
class Kernel(ASTNode):
    qualifier: str
    type: str
    name: str
    parameters: List["Parameter"]
    body: "Body"

@dataclass
class Parameter(ASTNode):
    type: str
    name: str

@dataclass
class Body(ASTNode):
    statements: List["Statement"]

# base class for declaration and assignment
class Statement(ASTNode):
    pass

@dataclass
class Declaration(Statement):
    type: str
    name: str
    value: Optional["Expression"] = None

@dataclass
class Assignment(Statement):
    name: str
    #index: str #
    value: "Expression" # instance of Expression class. "Expression" with quotes because Expression class is not yet defined

# base class for expressions
class Expression(ASTNode):
    pass

@dataclass
class Binary(Expression):
    op: str
    left: Expression #Union["Binary", "Literal", "Variable"] # or left: Expression
    right: Expression #Union["Binary", "Literal", "Variable"] # or right: Expression

@dataclass
class Literal(Expression): # constant
    value: Union[int, float]

@dataclass
class Variable(Expression): # var name
    name: str

# Transformer class
class CUDA_AST(Transformer):
    """ 
    Class where you define how each grammar rule (and parse_tree nodes) becomes an AST node. 
    You create a Transformer class with methods for each grammar rule. Each method takes the parse tree's 
    children (tokens or subtrees) and returns an AST node.
    """
    def kernel(self, items):
        qualifier, type, name, params, body = items
        return Kernel(qualifier=qualifier, type=type, name=name, parameters=params, body=body)

    def params(self, items):
        return items

    def parameter(self, items):
        type, name = items
        return Parameter(type=type, name=name) # "'list' object has no attribute 'value'". I took off the ".value"

    def body(self, block):
        return Body(statements=block)

    def declaration(self, items):
        type, name = items[:2]
        initializer = items[2:] if len(items) > 2 else None
        return Declaration(type=type, name=name, value=initializer)

    #def assignment(self, items):
    #    name, value = items
    #    return Assignment(name=name, value=value)

    def assignment(self, items):
       # if len(items) == 3:
       #     name, index, value = items
            #return Assignment(name=name, index=index, value=value)
       # else:
        name, value = items
        #index = None
        return Assignment(name=name, value=value) 
            

    def expression(self, items):
        #print(f"EXPRESSION items: {items}")
        if len(items) == 1: # single term
            return items[0]
        left = items[0]
        for op, right in zip(items[1::2], items[2::2]):
            left = Binary(op=op, left=left, right=right)
        #print(f"items:{items} left:{left}")
        return left

    def term(self, items):
        #print(f"TERM ITEMS: {items}")
        if len(items) == 1: # only one factor
            return items[0]
        left = items[0]     # multiple factors
        for op, right in zip(items[1::2], items[2::2]):
            left = Binary(op=op, left=left, right=right)
        return left

    def factor(self, items):
        print(f"FACTOR ITEMS: {items}")
        #if isinstance(items, Token): 
        if items[0].type == "NUMBER":
            #print(f"is number!")
            return Literal(value=items[0].value)
        elif items[0].type == "NAME":
            #print(f"is identifier!")
            return Variable(name=items[0].value)
        #elif items[0].type == ""
        return items[0]#.value

    def qualifier(self, token):
        return token[0].value

    def type(self, token):
        return token[0].value   
    
    #def base_type(self, token):
    #    return token[0].value

    def term_ops(self, token):
        #print(f"token {token}")
        return token[0].value

    def factor_ops(self, token):
        return token[0].value

    #def ops(self, token):
    #   return token[0].value
    
    def identifier(self, token):
        return token[0] #.value

    def array_index(self, items):
        print(f"items: {items}")
        return items   

# Expression is "Term((+|-) Term)*" . Ex: (8 + 24 * 2) is an expression because there are one Term '8' + another Term '24 * 2'
# Term is "Factor((*|/) Factor)*" Ex: (8) + (24 * 2), so 8 is a term and (24 * 2) is another term, because they are splitted by + signal
# Factor are the numbers. Ex: 8 + 24 * 2, so 8, 24, 2 are all factors

# after move this kernel to "kernel.cu"
kernel = r"""
__global__ void add(int a, int b) {
    int c = a + b;
}
"""

kernel_vecAdd = r"""
__global__ void vecAdd(int* a, int* b, int* c, int idx) {
    c[idx] = a[idx] + b[idx];   
}
"""

parser = Lark(cuda_grammar)
tree = parser.parse(kernel_vecAdd)
print(tree.pretty())

x = CUDA_AST()
ast = x.transform(tree)
print(ast.pretty())


# "binary" = 2 operands and 1 operator: 2 + 3 or 2 + (3 * 4) or a + b
# "literal" = constant value: 2 or 3.1 or "string"
# "variable" = a named identifier: a or result or threadIdx

# Obs: in ASTs, the interior nodes are the operators (+, *, -, /, ...) And the leaves are the operands (2, 6, 1, ...)