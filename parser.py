import os
from pathlib import Path
from lark import Lark

class Parser:
    """ CUDA parser using Lark """
    
    def __init__(self, cuda_path="./"):
        self.cuda_path = cuda_path
        self.cuda_version = self.get_cuda_version()
    
    def get_cuda_version(self):
        file = Path(self.cuda_path)
        if file.exists():
            return str(os.system("nvcc --version"))
        raise RuntimeError("CUDA instalation not found!")

    def cuda_parse(self, cuda_file, cuda_grammar, viz=0):
        try:
            with open(cuda_file, "r") as file:
                cuda_kernel = file.read()
            parser = Lark(cuda_grammar)
            tree = parser.parser(cuda_kernel)
            if viz == 1:
                print(tree.pretty())
                return tree
            else:
                print("The CUDA kernel has been parsed!")
                return tree
        
        except Exception as e:
            return f"Failed to parse {cuda_file} using the grammar!"





