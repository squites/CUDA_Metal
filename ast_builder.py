from lark import Transformer, Token
from dataclasses import dataclass
from typing import Union, List, Optional

# OBS: add parent attribute to make reference to the parent node of a node. This will help with tracking variables
# Or maybe I can pass the node as parent on the visit method, because we call visit for the children of that node. So we just need
# to pass the node itself as parent. Then we check the parent of that node

# classes for nodes of the Abstract Syntax Tree. So each class is an Object that will be a node when called.
class CUDA_Ast:
    def children(self):
        return []

    def pretty_print(self, indent=0):
        raise NotImplementedError

@dataclass
class CUDA_Program(CUDA_Ast):
    header: Optional[str]
    kernel: "Kernel"

    def pretty_print(self, indent=0):
        space = " " * indent
        print("Program(")
        print("library")
        print("kernel")
        print(")")

# semantic classes
@dataclass
class Kernel(CUDA_Ast):
    #parent: CUDA_Program
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
    mem_type: Optional[str]# = "__global__"
    #const: Optional[str] # need to treat when params are "const"
    type: str
    name: str
    #value: str

    #def pretty_print(self, indent):
    #    space = " " * indent
    #    print(f"{space}Parameter(type={self.type}, name={self.name}, mem_type={self.mem_type}, const={self.const})")

    def pretty_print(self, indent):
        space = " " * indent
        print(f"{space}Parameter(mem_type={self.mem_type}, type={self.type}, name={self.name})")


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
    memory: Optional[str]# = None
    type: str
    name: str
    value: Optional["Expression"] = None
    #parent: Node

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
    # parent: Node

    def pretty_print(self, indent=0):
        space = " " * (indent+2)
        print(f"    Assignment(")
        print(f"{space}name={self.name},")
        print(f"{space}value={self.value}")
        print("    )")

@dataclass
class IfStatement(Statement):
    condition: "Expression" # expression
    if_body: List[Statement] # statement*
    # parent: Node

    def children(self):
        return [*self.if_body] #if self.if_body is not None else []

    def pretty_print(self, indent=0):
        space = " " * (indent+2)
        print("    IfStatement(")
        print(f"{space*2}condition={self.condition},")
        print(f"{space*2}if_body={self.if_body}")
        print("    )")

@dataclass
class ForStatement(Statement):
    init: Declaration
    condition: "Expression"
    increment: Assignment
    forBody: List[Statement]
    #parent: Union[Statement]

    def children(self):
        return [self.forBody]
    
    def pretty_print(self, indent=0):
        space = " " * (indent+2)
        print("    ForStatement(")
        print(f"{space*2}initialization={self.initialization},")
        print(f"{space*2}condition={self.condition},")
        print(f"{space*2}increment={self.increment},")
        print(f"{space*2}forBody={self.forBody}")
        print("    )")


# base class for expressions. Base classes define a common type
class Expression(CUDA_Ast):
    pass

@dataclass
class Binary(Expression):
    op: str
    left: Expression 
    right: Expression
    #parent: Union[Statement, Expression]

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
    # parent: Node

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Literal(")
        print(f"{space}value={self.value}")
        print(")")

@dataclass
class Variable(Expression): # var name
    name: str
    #parent: Node

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}Variable(")
        print(f"{space}name={self.name}")
        print(")")

@dataclass
class Array(Expression):
    name: Variable
    index: Expression
    # parent: Node

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
    # parent: Node

    def pretty_print(self, indent=0):
        space = " " * indent
        print(f"{space}CudaVar(")
        print(f"{space}base={self.base},")
        print(f"{space}dim={self.dim}")
        print(")")

@dataclass
class ThreadId:
    dim: str

    def pretty_print(self, indent=0):
        raise NotImplementedError 

# Transformer class
class CUDATransformer(Transformer):
    """ 
    Class where you define how each grammar rule (and parse_tree nodes) becomes an AST node. 
    You create a Transformer class with methods for each grammar rule. When you call ".transform()", each method 
    takes the parse tree's children (tokens or subtrees) and returns an AST node.
    """
    def start(self, items):
        return items[0]
    
    def program(self, items):
        if len(items) > 1:
            header, kernel = items
        else:
            header = None
            kernel = items[0]
        return CUDA_Program(header=header, kernel=kernel)

    def kernel(self, items):
        qualifier, name, params, body = items
        return Kernel(qualifier=qualifier, type="void", name=str(name), parameters=params, body=body)

    def params(self, items):
        return items

    def parameter(self, items):
        mem_type = "__global__"
        if len(items) == 3 and str(items[0]) == "__shared__" or str(items[0]) == "__constant__":
            mem_type = str(items[0])
            type = str(items[1])
            name = str(items[2])
        elif str(items[0]) == "const" or str(items[1]) == "const":
            # add constant here eventually
            type = str(items[1]) if len(items) == 3 else str(items[2])
            name = str(items[2]) if len(items) == 3 else str(items[3])
        else:
            type = str(items[0])
            name = str(items[1])
        return Parameter(mem_type=mem_type, type=type, name=name)
        
    def body(self, block):
        return Body(statements=block)

    # Used to replace "Tree(Token('RULE', 'statement'), ...)"
    def statement(self, items):
        return items[0]
    
    def declaration(self, items):
        if items[0] == "__shared__" or items[0] == "__constant__":
            memory = items[0]
            type = items[1]
            name = items[2]
            initializer = items[3] if len(items) == 4 else None
        else:
            memory = None
            type = items[0]
            name = items[1]
            initializer = items[2] if len(items) == 3 else None

        return Declaration(memory=str(memory), type=str(type), name=name, value=initializer)

    def assignment(self, items): # error! for some reason the name is returning as TOKEN instead of the string 
        name = str(items[0]) if isinstance(items[0], Token) else items[0]
        value = items[1]
        return Assignment(name=name, value=value)
        
    def if_statement(self, items):
        # items[0]: if (x < M && y < N) {
        # items[1]:     float tmp = 0.0;
        # items[2:      for (int i = 0; i < K; i=i+1) { tmp = tmp + (data0[x * K + i] * data1[i * N + y]);}
        # items[3]:     data2[x * N + y] = tmp;
        condition = items[0]
        if_body = items[1:]
        return IfStatement(condition=condition, if_body=if_body)

    # obs: should we treat parameter variables as Parameter() or Variable() node. Because here the condition 
    # is based on a Parameter, but we generate as a Variable node
    def for_statement(self, items):
        init = items[0]
        cond = items[1]
        incr = items[2]
        forBody = items[3:]
        #init, cond, incr, forBody = items
        return ForStatement(init=init, condition=cond, increment=incr, forBody=forBody)

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

    def logical_ops(self, token):
        return token[0].value
    
    def identifier(self, token):
        return token[0].value #.value

    def memory_type(self, token):
        return token[0].value

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
class METAL_Program(METAL_Ast):
    header: str
    kernel: "METAL_Kernel"

@dataclass
class METAL_Kernel(METAL_Ast):
    qualifier: str
    type: str
    name: str
    parameters: List["METAL_Parameter"]
    body: "METAL_Body"

    def children(self):
        return [*self.parameters, *self.body] # dont know why but I added '*' before self.body, and the generator started to understand this as METAL_Body node instead of list.

@dataclass
class METAL_Parameter(METAL_Ast):
    memory_type: str #= None
    #constant: str = ""
    type: str
    name: str
    buffer: Optional[str] # for some parameters we use this buffer
    init: Optional[Union[str, int]] = None

@dataclass
class METAL_Body(METAL_Ast):
    statements: List["METAL_Statement"]

    def children(self):
        return [*self.statements]

# base class for statements
class METAL_Statement(METAL_Ast):
    pass

@dataclass
class METAL_Declaration(METAL_Statement):
    memory: Optional[str]
    type: str
    name: str
    value: Optional["METAL_Expression"] = None # is generating a list value=[METAL_Binary(...)] while in cuda_ast generates value=Binary(...)

    def children(self):
        return [*self.value] if self.value is not None else None # when adding '*', removes the "[]" for some reason

@dataclass
class METAL_Assignment(METAL_Statement):
    name: str
    value: "METAL_Expression" # forward reference

@dataclass
class METAL_IfStatement(METAL_Statement):
    condition: "METAL_Expression"
    if_body: List[METAL_Statement]

    def children(self):
        return [*self.if_body]

@dataclass
class METAL_ForStatement(METAL_Statement):
    init: METAL_Declaration
    condition: "METAL_Expression"
    increment: METAL_Assignment
    forBody: List[METAL_Statement]
    parent: Union[METAL_Statement, "METAL_Expression"]

    def children(self):
        return [*self.forBody]

# base class for expressions
class METAL_Expression(METAL_Ast):
    pass

@dataclass
class METAL_Binary(METAL_Expression):
    op: str
    left: METAL_Expression
    right: METAL_Expression

@dataclass
class METAL_Literal(METAL_Expression):
    value: Union[int, float]

@dataclass
class METAL_Variable(METAL_Expression):
    name: str

@dataclass
class METAL_Array(METAL_Expression):
    name: METAL_Variable
    index: METAL_Expression

@dataclass
class METAL_Var(METAL_Ast):
    metal_var: str

@dataclass
class METAL_GlobalThreadId(METAL_Ast):
    def __init__(self):
        raise NotImplementedError
    #pass

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
# 1) - data is a Token('RULE', 'qualifier'), which means that the rule_name is 'qualifier'
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