# CUDAMetal

CUDAMetal is a transpiler that translates CUDA kernels into METAL Shaders.

***Note: This project is still under development.***

## Introduction
This project is intended to learn about how transpilers work and to be able to run CUDA kernels on my MacBook M1. This is also a way to compare performance on two different GPUs and see how the same optimization is done on two different machines. This work is inspired by https://github.com/thomaschlt/CUDApple. This project also uses knowledge from Simon's blog: https://siboehm.com/articles/22/CUDA-MMM.

## How it works
The translation works as:
1. Parse Tree: CUDA source code is parsed into a parse tree using Lark's library that helps to create a tree based on a defined grammar, which is then walked by Lark's Transformer generating CUDA AST nodes based on their rules. (``` ./ast_builder.py```)
2. AST: After the first AST, we then traverse the tree generating a METAL AST, converting CUDA nodes into METAL nodes ``` ./traverse.py```. During this process, the transpiler performs the compiler optimizations.
3. Code Generation: CodeGen traverse the METAL AST and generates corresponded indented METAL Shadder code string. (``` ./codegen.py```). But also, the transpiler generates a JSON file that stores kernel's metadata. This information is essential to generate automatically the kernel dispatcher code as well.

## Execution
To run the transpiler the user needs to pass the CUDA kernel file path, grid and block sizes and the data size. 
```bash
python3 main.py "./examples/naive_matmul.cu" --grid=1,1,1 --block=8,8,1 -N=64
```
The program will generate mainly `naive_matmul.metal`, `dispatcher.mm`, `input.bin` and `output.bin`. 

## Optimization
The transpiler currently performs canonicalization on the terms of the expression and apply constant folding automatically. Eventually, there will be loop optimizations, constant propagations, and many more.

## TODO
- [x] Vector Addition
- [x] Naive Matmul
- [ ] GEMM with Shared Memory Cache-Blocking
- [ ] GEMM 2D Blocktiling
- [ ] GEMM with vectorized Shared Memory and Global Memory accesses
- [ ] Softmax
- [ ] 2D Convolution
- [ ] Training simple MLP
- [ ] Attention
- [ ] Flash Attention
- [ ] Replace Lark's use to my own lexer/parser