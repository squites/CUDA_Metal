# CUDAMetal

CUDAMetal is a transpiler that translates CUDA kernels into METAL Shaders.

***Note: This project is still under development.***

## Introduction
This project is intended to learn about how transpilers work and to be able to run CUDA kernels on my MacBook M1. This is also a way to compare performance on two different GPUs and see how the same optimization is done on two different machines. This project is inspired by https://github.com/thomaschlt/CUDApple. This project also uses knowledge from Simon's blog: https://siboehm.com/articles/22/CUDA-MMM.

## How it works
The translation works as:
1. Parse Tree: CUDA source code is parsed into a parse tree using Lark's library that helps to create a tree based on a grammar, which is then walked by Lark's Transformer generating CUDA AST nodes based on their rules. (``` ./ast_builder.py```)
2. AST: After the first AST, we traverse the tree generating a METAL AST, converting CUDA nodes into METAL nodes ``` ./traverse.py```
3. Code Generation: CodeGen traverse the METAL AST and generates corresponded indented METAL Shadder code string. (``` ./codegen.py```)

## Optimization
The transpiler currently performs canonicalization on the terms of the expression and apply constant folding (not done yet) automatically. Eventually, there will be a tree normalization to simplify nodes of the AST.

## TODO
- [x] Vector Addition
- [x] Naive Matmul
- [ ] GEMM with Shared Memory Cache-Blocking
- [ ] GEMM 2D Blocktiling
- [ ] GEMM with vectorized Shared Memory and Global Memory accesses
- [ ] Softmax
- [ ] 2D Convolution
- [ ] Attention
- [ ] Complete Training