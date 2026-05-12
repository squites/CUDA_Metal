from lark import Lark

cuda_grammar = r"""
    #start: kernel*
    start: program
    program: ("\#include\<"NAME"\>")? kernel*
    #library: HASH "include" "\<" NAME "\>"
    #library: NAME 

    # kernel signature
    kernel: qualifier "void" identifier "(" params ")" "{" body "}"
    params: [parameter ("," parameter)*]
    parameter: memory_type? "const"? type identifier
    body: statement*
    
    statement: declaration ";"
             | assignment  ";"
             | if_statement 
             | for_statement
             | syncthreads ";"
             | atomic_statement ";"
             | increment ";"

    if_statement: "if (" expression ") {" statement* "}" #("else {" statement* "}")?
    for_statement: "for (" (declaration | assignment) "; " expression "; " (assignment| increment)") {" statement* "}"
    atomic_statement: atomics "(" "&"? expression "," expression ("," expression)* ")"
    
    # barrier
    syncthreads: "__syncthreads()"

    # statements
    declaration: memory_type? type (identifier|array_index) ("=" (expression|atomic_statement))?
    #assignment: (array_index | identifier) "=" expression
    assignment: (array_index | identifier) ASSIGN_OP expression # add "increment"
    expression: term ((term_ops | logical_ops) term)*

    term: factor (factor_ops factor)*
    
    factor: NUMBER
          | "-"NUMBER # negative
          | identifier
          | "(" expression ")"
          | array_index
          | cuda_var
          | func_call
    
    # needed to AST
    qualifier: QUALIFIER
    type: TYPE
    term_ops: TERM_OPS
    factor_ops: FACTOR_OPS
    logical_ops: LOGICAL_OPS
    identifier: NAME
    array_index: identifier ("[" expression "]") # a[i+1]
    memory_type: MEM_TYPE
    atomics: ATOMIC_FUNC
    func_call: FUNC_NAME "(" (expression ("," expression)*)? ")"
    increment: (identifier | array_index) INCREMENT_OP
    
    # types and ops
    QUALIFIER: "__global__" | "__device__" | "__host__"
    TYPE: /(int|float|void)\*?/
    TERM_OPS: "+" | "-"
    FACTOR_OPS: "*" | "/" | "%"
    LOGICAL_OPS: "==" | ">" | "<" | ">=" | "<=" | "!=" | "&&"
    ASSIGN_OP: "=" | "+=" | "/=" | "*=" 
    INCREMENT_OP: "++" | "--"
    cuda_var: BASE_VAR "." CUDA_DIM
    BASE_VAR: ("blockIdx" | "blockDim" | "threadIdx")
    CUDA_DIM: ("x" | "y" | "z")
    MEM_TYPE: "__shared__" | "__constant__"
    ATOMIC_FUNC: "atomicAdd" | "atomicSub" | "atomicCAS"
    FUNC_NAME: "expf" | "sqrtf"

    # imports 
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""