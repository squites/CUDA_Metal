from lark import Lark, Transformer

with open("kernel.cu", 'r') as kernel:
    kernel = kernel.read()

cuda_grammar = r"""
    start: kernel

    kernel: type_specifier NAME "(" params_list ")" "{" statement* "}"
    
    type_specifier: KEYWORD? TYPE pointer*
    pointer: "*"
    
    params_list: (param ("," param)*)?
    param: type_specifier NAME
    
    statement: index
             | declaration
             | operation
             | if
    
    index: type_specifier NAME "=" CUDA_VAR "*" CUDA_VAR "+" CUDA_VAR ";" 
    declaration: type_specifier NAME "=" NUMBER ";"
    operation: NAME "[" NAME "]" "=" NAME "[" NAME "]" "+" NAME "[" NAME "]" ";"   
    if: "if" "(" cond ")" "{" statement* "}"
    cond: (NAME (logical_op NAME)*)+
    logical_op: "<"
              | ">"
              | "<="
              | ">="
    
    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /\d+/
    KEYWORD: "__global__"
           | "__device__"   
    TYPE: "void"
        | "int"
        | "float" 

    CUDA_VAR: "blockIdx.x"
            | "blockIdx.y"
            | "blockIdx.z"
            | "blockDim.x"
            | "blockDim.y"
            | "blockDim.z"
            | "threadIdx.x"
            | "threadIdx.y"
            | "threadIdx.z"
    
    %import common.WS
    %ignore WS
"""

parser = Lark(cuda_grammar)
tree = parser.parse(kernel)
print(tree.pretty())
