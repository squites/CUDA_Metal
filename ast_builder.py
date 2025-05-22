from lark import Lark, Transformer, Token
#from parser import tree
from dataclasses import dataclass
from typing import Union, List, Optional
from enum import Enum

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
# The transformer is where you define how each grammar rule (and its parse_tree nodes) becomes an AST node. 
# You create a Transformer class with methods for each grammar rule. Each method takes the parse tree?s 
# children (tokens or subtrees) and returns an AST node.

class CUDA_AST(Transformer):
    
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

    def assignment(self, items):
        name, value = items
        return Assignment(name=name, value=value)

    def expression(self, items):
        if len(items) == 1: # single term
            return items[0]
        left = items[0]
        for op, right in zip(items[1::2], items[2::]):
            left = Binary(op=op, left=left, right=right)
        return left

    def term(self, items):
        if len(items) == 1: # only one factor
            return items[0]
        left = items[0]     # multiple factors
        for i in items:
            op = items[1::2]
            right = items[2::2]
            return Binary(op=op, left=left, right=right)
        #for op, right in zip(items[1::2], items[2::2]):
        #    left = Binary(op=op, left=left, right=right)
        #return left

    def factor(self, items):
        #item = items[0]
        if isinstance(items, Token):
            if items[0].type == "NUMBER":
                return Literal(value=items[0].value)
            elif items[0].type == "identifier":
                return Variable(name=items[0].value)
            return items[0].value

    def qualifier(self, token):
        return token[0].value

    def type(self, token):
        return token[0].value

    def ops(self, token):
        return token[0].value
    
    def identifier(self, token):
        return token[0].value


grammar = r"""
    start: kernel*

    kernel: qualifier type identifier "(" params ")" "{" body "}"
    params: [parameter ("," parameter)*]
    parameter: type identifier
    # params: (declaration ("," declaration)*)?
    body: statement*
    
    statement: declaration ";"
             | assignment  ";"

    declaration: type identifier ("=" expression)? # var declaration
    assignment: identifier "=" expression 
    expression: term (ops term)*

    term: factor (ops factor)*
    
    factor: NUMBER
          | identifier
          | "(" expression ")"
    
    qualifier: QUALIFIER
    type: TYPE
    ops: OPS
    #term_ops: "+" | "-"
    #factor_ops: "*" | "/"
    identifier: NAME

    # added these so Tranformer can call the methods related
    QUALIFIER: "__global__" | "__device__" | "__host__"
    TYPE: "void" | "int" | "float"
    OPS: "+" | "*" | "-" | "/" 

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""
# Expression is "Term((+|-) Term)*" . Ex: (8 + 24 * 2) is an expression because there are one Term '8' + another Term '24 * 2'
# Term is "Factor((*|/) Factor)*" Ex: (8) + (24 * 2), so 8 is a term and (24 * 2) is another term, because they are splitted by + signal
# Factor are the numbers. Ex: 8 + 24 * 2, so 8, 24, 2 are all factors

code = r"""
__global__ void add(int a, int b) {
    int c = a + b;
    int i = 5;
}
"""

parser = Lark(grammar)
tree = parser.parse(code)
print(tree.pretty())

x = CUDA_AST()
ast = x.transform(tree)
print(ast)


# "binary" = 2 operands and 1 operator: 2 + 3 or 2 + (3 * 4) or a + b
# "literal" = constant value: 2 or 3.1 or "string"
# "variable" = a named identifier: a or result or threadIdx

# Obs: in ASTs, the interior nodes are the operators (+, *, -, /, ...) And the leaves are the operands (2, 6, 1, ...)
# We represent AST nodes for the expression "7 + 3 * 4" as: 
# {
#   type: "BinExpr", 
#   op: "+", 
#   left: {
#       type: "NumericLiteral", 
#       value: 7
#   },
#   right: {
#       type: "BinExpr",
#       op: "*",
#       left: {
#           type: "NumericLiteral",
#           value: 3,
#       },
#       right: {
#           type: "NumericLiteral",
#           value: 4,
#       }
#   } 
# }

"""
# This is a test example to see how the AST class are!
class AST(Transformer):
    def __init__(self):
        super().__init__()

    def kernel(self, tokens):
        type = tokens[0]
        name = tokens[1]
        params = tokens[2] 
        body = tokens[3:]
        return Kernel(type, name, params, body)

    def params(self, items):
        return items # list of declarations

    def declaration(self, tokens):
        type = tokens[0]
        name = tokens[1]
        val = tokens[2] if len(tokens) > 2 else None
        return Declaration(type, name, val) #Declaration(type, name, val)

    def assignment(self, tokens):
        identifier = tokens[0]
        expr = tokens[1]
        return Assignment(identifier, expr)

    def expression(self, tokens):
        left = tokens[0]
        for i in range(1, len(tokens), 2):
            op = tokens[i]
            right = tokens[i+1]
            left = Expression(op, left, right)
        return left
    
    def term(self, tokens):
        fact_left, op, fact_right = tokens
        
        factor_l = tokens[0]
        for i in range(1, len(tokens), 2):
            op = token[i]
            factor_r = token[i+1]
            factor_l = Expression(op, factor_l, factor_r)
        return factor_l

    def factor(self, token):
        return token

    def binary(self, tokens):
        left, op, right = tokens
        return Binary(op, left, right)
    
    def literal(self, value):
        return Literal(value)

    def variable(self, name):
        return Variable(name)

    def identifier(self, tokens):
        return tokens[0].value

    def type(self, tokens):
        return str(tokens)

    def NUMBER(self, items):
        return int(items.value)
"""
