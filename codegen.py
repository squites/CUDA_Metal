from main import METAL_Ast

class CodeGen():
    """
    This will walk the METAL ast and generate a string which is the metal code 
    """
    def __init__(self, metal_ast):
        self.metal_ast = metal_ast

    