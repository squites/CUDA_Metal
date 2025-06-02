from lark import Lark, Transformer, Token, Tree
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

# maybe create a new class ArrayAccess
@dataclass
class Array(Expression):
    name: Variable
    index: Expression

@dataclass
class CudaVar:
    base: str
    dim: str

# Transformer class
class CUDA_AST(Transformer):
    """ 
    Class where you define how each grammar rule (and parse_tree nodes) becomes an AST node. 
    You create a Transformer class with methods for each grammar rule. Each method takes the parse tree's 
    children (tokens or subtrees) and returns an AST node.
    """
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

    def declaration(self, items):
        type, name = items[:2]
        initializer = items[2:] if len(items) > 2 else None
        return Declaration(type=type, name=str(name), value=initializer)

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
            #print(f"FACTOR TOKEN: {items}   items[0]: {items[0].value}")
            if item.type == "NUMBER":
                return Literal(value=item.value)
            elif item.type == "NAME":
                return Variable(name=item.value)
        #elif isinstance(item, Tree):
        #    print(f"FACTOR TREE: {items}   items[0]: {items[0]}")
        #    if item.data == "cuda_var":
        #        return CudaVar(var=item, dim=item) 
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
print(tree.pretty())

x = CUDA_AST()
ast = x.transform(tree)
print(ast.pretty())


# "binary" = 2 operands and 1 operator: 2 + 3 or 2 + (3 * 4) or a + b
# "literal" = constant value: 2 or 3.1 or "string"
# "variable" = a named identifier: a or result or threadIdx

# Obs: in ASTs, the interior nodes are the operators (+, *, -, /, ...) And the leaves are the operands (2, 6, 1, ...)
# Expression is "Term((+|-) Term)*" . Ex: (8 + 24 * 2) is an expression because there are one Term '8' + another Term '24 * 2'
# Term is "Factor((*|/) Factor)*" Ex: (8) + (24 * 2), so 8 is a term and (24 * 2) is another term, because they are splitted by + signal
# Factor are the numbers. Ex: 8 + 24 * 2, so 8, 24, 2 are all factors
