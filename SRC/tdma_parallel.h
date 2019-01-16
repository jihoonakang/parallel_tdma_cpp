#ifndef TDMA_PARALLEL_H
#define TDMA_PARALLEL_H

/**
 * @brief       Parallel tri-diagonal matrix solver using cyclic reduction (CR), parallel CR (PCR),
 *              and Thomas+PCR hybrid algorithm
 * @details     The CR algorithm is described on Parallel Scientific Computing in C++ and MPI
 *              by Karniadakis and Kirby. CR algorithm removes odd rows recursively,
 *              so MPI processes begin to drop out after single row is left per MPI process,
 *              while PCR can use full parallelism. Therefore, PCR is a good solution from
 *              the level where single row is left per MPI process. In this implementation,
 *              we can choose CR or PCR algorithm from the single-row level.
 *              Odd-rows are removed successively and we obtain two reduced equations finally.
 *              Obtained solutions from 2x2 matrix equations are used to obtain other unknowns.
 *              Hybrid Thomas-PCR algorithm is from the work of Laszlo, Gilles and Appleyard, 
 *              Manycore Algorithms for Batch Scalar and Block Tridiagonal Solvers, ACM TOMS, 
 *              42, 31 (2016).
 *
 * @author      Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
 * @date        20 January 2019
 * @version     0.1
 * @par         Copyright
                Copyright (c) 2019 by Ji-Hoon Kang. All rights reserved.
 * @par         License     
                This project is release under the terms of the MIT License (see LICENSE in )
*/

class tdma_parallel {

    private :
        /// Number of rows per MPI process and should be 2^n.
        int n_mpi;
        
        /// Number of MPI process and should be also 2^m.
        int nprocs;

        /// MPI process ID
        int myrank;

        /// Local private pointer for coefficient maxtix a
        double *a;
        /// Local private pointer for coefficient maxtix b
        double *b;
        /// Local private pointer for coefficient maxtix c
        double *c;
        /// Local private pointer for RHS vector r
        double *r;
        /// Local private pointer for solution vector x
        double *x;

        void cr_forward_multiple_row();
        void cr_backward_multiple_row();
        void cr_forward_single_row();
        void cr_backward_single_row();
        void pcr_forward_single_row();

        void pThomas_forward_multiple_row();
        void pcr_double_row_substitution();

    public :

        tdma_parallel();
        ~tdma_parallel();

        void setup(int n, int np_world, int rank_world);
        void cr_solver    (double *a_mpi, double *b_mpi, double *c_mpi, double *r_mpi, double *x_mpi);
        void cr_pcr_solver(double *a_mpi, double *b_mpi, double *c_mpi, double *r_mpi, double *x_mpi);
        void Thomas_pcr_solver(double *a_mpi, double *b_mpi, double *c_mpi, double *r_mpi, double *x_mpi);
        void verify_solution(double *a_ver, double *b_ver, double *c_ver, double *r_ver, double *x_sol);

};

#endif