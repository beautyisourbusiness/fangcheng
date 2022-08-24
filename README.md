# FANGCHENG

Fangcheng is a pure-Cairo **solver** of systems of linear equations in fixed-point arithmetic. It currently uses naive methods such as Gauss-Jordan, and also provides a set of **primitives** for elementary matrix operations. It will grow gradually, and we will incorporate in the future more primitives and faster and cheaper methods of resolution.

**V0.1**(2022-08-24): Working solver of linear systems. Facilities for importing and processing matrices. 

**Future work**

Features: More primitives and std matrix functions will be supported like determinants, eigenvalues, calc of condition numbers, basic lp, etc. Compatibility tests. Cheaper and faster decomposition methods benchmarking. 

Internals: integrated testing, refactor code to improve readability, more std lib structure

To view a sample demo of resolution, you can:

`cairo-compile matrix_test.cairo --output test.json`


and then run it with a sample augmented matrix that represents your system of choice, like:

`cairo-run --program=test.json --layout=all --program_input=samplesystems/augm_matrix3x4.json`


you’ll see the input matrix, the canonical/echelon form matrix and the set of solutions.


why the name?: Fancheng (something like square arrays) is the name given by ancient Chinese mathematicians to the procedure for solving systems of linear equations, essentially equivalent to modern methods of Gaussian elimination.

Actual scope: This project is part of a general research on compact numerical methods in acccordance to Cairo architecture constraints.

#WARNING# This repository is a work in progress, don't use it in production unless explicitly mentioned in the code!

BIOB 
