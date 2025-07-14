from lark import Lark

cuda_grammar = r"""
    start: kernel*

    # kernel signature
    kernel: qualifier "void" identifier "(" params ")" "{" body "}"
    params: [parameter ("," parameter)*]
    parameter: memory_type? type identifier # added memory_type
    body: statement*
    
    statement: declaration ";"
             | assignment  ";"

    # statements
    declaration: type identifier ("=" expression)? # var declaration 
    assignment: (array_index | identifier) "=" expression
    expression: term (term_ops term)*

    term: factor (factor_ops factor)*
    
    factor: NUMBER
          | identifier
          | "(" expression ")"
          | array_index
          | cuda_var
    
    # needed to AST
    qualifier: QUALIFIER
    type: TYPE
    term_ops: TERM_OPS
    factor_ops: FACTOR_OPS
    identifier: NAME
    array_index: identifier ("[" expression "]") # a[i+1]
    memory_type: MEM_TYPE
    
    # types and ops
    QUALIFIER: "__global__" | "__device__" | "__host__"
    TYPE: /int\*?|float\*?|void\*?/ 
    #TYPE: BASE_TYPE "*"?
    #BASE_TYPE: "void" | "int" | "float"
    TERM_OPS: "+" | "-"
    FACTOR_OPS: "*" | "/"
    #cuda_var: ("blockIdx" | "blockDim" | "threadIdx") "." ("x" | "y")
    cuda_var: BASE_VAR "." CUDA_DIM
    BASE_VAR: ("blockIdx" | "blockDim" | "threadIdx")
    CUDA_DIM: ("x" | "y")
    MEM_TYPE: "__shared__" | "__constant__"

    # imports 
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""