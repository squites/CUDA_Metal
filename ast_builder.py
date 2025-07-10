from lark import Transformer, Token
from dataclasses import dataclass
from typing import Union, List, Optional

# classes for nodes of the Abstract Syntax Tree. So each class is an Object that will be a node when called.
class CUDA_Ast:
    def children(self):
        return []

    def pretty_print(self, indent=0):
        raise NotImplementedError

# semantic classes
@dataclass
class Kernel(CUDA_Ast):
    qualifier: str
    type: str
    name: str
    parameters: List["Parameter"]
    body: "Body"

    def children(self):
        return [*self.parameters, self.body]

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
class Parameter(CUDA_Ast):
    type: str
    name: str

    def pretty_print(self, indent):
        space = " " * indent
        print(f"{space}Parameter(type={self.type}, name={self.name})")

@dataclass
class Body(CUDA_Ast):
    statements: List["Statement"]

    def children(self):
        return [*self.statements]

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Body(statements=[")
        for stmnt in self.statements:
            stmnt.pretty_print(indent+2)
        print(f"{space}])")

# base class for declaration and assignment
class Statement(CUDA_Ast):
    pass

@dataclass
class Declaration(Statement):
    type: str
    name: str
    value: Optional["Expression"] = None

    def children(self):
        return [self.value] if self.value is not None else []

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

# base class for expressions. Base classes define a common type
class Expression(CUDA_Ast):
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
class CUDATransformer(Transformer):
    """ 
    Class where you define how each grammar rule (and parse_tree nodes) becomes an AST node. 
    You create a Transformer class with methods for each grammar rule. When you call ".transform()", each method 
    takes the parse tree's children (tokens or subtrees) and returns an AST node.
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
        print(f"value type: {type(value)}\n value:{value}")
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
        return str(items[0].value)

    def cuda_dim(self, items):
        return str(items[0].value)
        
    def cuda_var(self, items):
        base, dim = items
        return CudaVar(base=str(base), dim=str(dim))


################# METAL ###################
class METAL_Ast():
    def children():
        return []

@dataclass
class METAL_Kernel_node(METAL_Ast):
    qualifier: str
    type: str
    name: str
    parameters: List["METAL_parameter_node"]
    body: "METAL_body_node"

    def children(self):
        return [*self.parameters, self.body]

@dataclass
class METAL_Parameter_node(METAL_Ast):
    memory_type: str
    type: str
    name: str

@dataclass
class METAL_Body_node(METAL_Ast):
    statements: List["METAL_statement_node"]

    def children(self):
        return [*self.statements]

# base class for statements
class METAL_Statement_node(METAL_Ast):
    pass

@dataclass
class METAL_Declaration_node(METAL_Statement_node):
    type: str
    name: str
    value: Optional["METAL_expression_node"] = None

@dataclass
class METAL_Assignment_node(METAL_Statement_node):
    name: str
    value: "METAL_Expression_node" # forward reference

# base class for expressions
class METAL_Expression_node(METAL_Ast):
    pass

@dataclass
class METAL_Binary_node(METAL_Expression_node):
    op: str
    left: METAL_Expression_node
    right: METAL_Expression_node

@dataclass
class METAL_Literal_node(METAL_Expression_node):
    value: Union[int, float]

@dataclass
class METAL_Variable_node(METAL_Expression_node):
    name: str

@dataclass
class METAL_Array_node(METAL_Ast):
    name: METAL_Variable_node
    index: METAL_Expression_node

@dataclass
class METAL_Var_node(METAL_Ast):
    metal_var: str



# ------------------------------ DOC ---------------------------------
# 
# "binary" = 2 operands and 1 operator: 2 + 3 or 2 + (3 * 4) or a + b
# "literal" = constant value: 2 or 3.1 or "string"
# "variable" = a named identifier: a or result or threadIdx

# Expression is "Term((+|-) Term)*" . Ex: (8 + 24 * 2) is an expression because there are one Term '8' + another Term '24 * 2'
# Term is "Factor((*|/) Factor)*" Ex: (8) + (24 * 2), so 8 is a term and (24 * 2) is another term, because they are splitted by + signal
# Factor are the numbers. Ex: 8 + 24 * 2, so 8, 24, 2 are all factors


# Transformer funcionality:
# Parse tree only has Tree() or Token().:
#   - Tree(data=rule_name, children=[...]) and
#   - Token(type=TERMINAL_SYMBOL, value)
#
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
#      so: def qualifier(self, ["__global__"])
# 
# 4) The qualifier method receives ["__global__"]:
#    - then it can create an AST node calling the corresponded class "Qualifier(...)"  or simply returns the string.
#
# When to return a string and when to create a AST node?
# Node: when represents a semantic structure. is not just a string, but a construct with meaning and behavior
# string: when is just a primitive value like: "int", 'x', "__global__", ...
#
# structure:
# Parse Tree:
#   Token(type="TYPE_NAME", value="actual_text")
#   Tree(data="rule_name", children=[...]) # data: str, children: list of Tree()s and Token()s
#
# AST:
#   - for every Tree(data="rule", children=[...]), Lark finds the method called "def rule(self, children)"
#     This method transform the parse_tree node into python object - AST node.