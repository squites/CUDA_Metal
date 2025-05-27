from lark import Lark

cuda_grammar = r"""
    start: kernel*

    kernel: qualifier type identifier "(" params ")" "{" body "}"
    params: [parameter ("," parameter)*]
    parameter: type identifier
    body: statement*
    
    statement: declaration ";"
             | assignment  ";"

    declaration: type identifier ("=" expression)? # var declaration
    #assignment: identifier "=" expression 
    assignment: (array_index | identifier) "=" expression
    expression: term (term_ops term)*

    term: factor (factor_ops factor)*
    
    factor: NUMBER
          | identifier
          | "(" expression ")"
          | array_index
    
    qualifier: QUALIFIER
    type: TYPE
    term_ops: TERM_OPS
    factor_ops: FACTOR_OPS
    identifier: NAME
    array_index: identifier ("[" expression "]") # a[i+1]
    
    # added these so Tranformer can call the methods related
    QUALIFIER: "__global__" | "__device__" | "__host__"
    TYPE: BASE_TYPE "*"?
    BASE_TYPE: "void" | "int" | "float"
    TERM_OPS: "+" | "-"
    FACTOR_OPS: "*" | "/"
    # improving the grammar
    # cuda_var: ("blockIdx" | "blockDim" | "threadIdx") "." ("x" | "y")

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""