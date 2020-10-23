# Parallel tri-diagonal matrix solver using cyclic reduction (CR), parallel CR (PCR), and Thomas+PCR hybrid algorithm

## Author
Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information

## Overview
The CR algorithm is described on Parallel Scientific Computing in C++ and MPI
by Karniadakis and Kirby. CR algorithm removes odd rows recursively,
so MPI processes begin to drop out after single row is left per MPI process,
while PCR can use full parallelism. Therefore, PCR is a good solution from
the level where single row is left per MPI process. In this implementation,
we can choose CR or PCR algorithm from the level where each MPI process has 
a single row.

Odd-rows are removed successively and we obtain two reduced equations finally,
from which the last-level solutions are obtained. The last-level solutions 
are back-substituted for all rows.

Hybrid Thomas-PCR algorithm is from the work of Laszlo, Gilles and Appleyard, 
Manycore Algorithms for Batch Scalar and Block Tridiagonal Solvers, ACM TOMS, 
42, 31 (2016).

This solver has restrictions on input parameters in which the number of rows
is 2^n and the number of MPI processes is 2^m, respectively, and n >= 2*m.

## Cite
Please us the folowing bibtex, when you refer to this project.

    @misc{kang2019ptdma,
        title  = {Parallel tri-diagonal matrix solver using cyclic reduction (CR), parallel CR (PCR), and Thomas+PCR hybrid algorithm},
        author = {Kang, Ji-Hoon},
        url    = https://github.com/jihoonakang/parallel_tdma_cpp},
        year   = {2019}
    }
