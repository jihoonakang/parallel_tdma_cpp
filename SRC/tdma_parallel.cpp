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

#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <algorithm>
#include "tdma_parallel.h"

using namespace std;

tdma_parallel :: tdma_parallel() {}
tdma_parallel :: ~tdma_parallel() {}

/** 
 * @brief   Initialize local private variables from global input parameters.
 * @param   n Size of global array
 * @param   np_world Number of MPI process
 * @param   rank_world rank ID in MPI_COMM_WORLD
*/
void tdma_parallel :: setup(int n, int np_world, int rank_world)
{
    nprocs = np_world;
    myrank = rank_world;
    n_mpi = n / nprocs;
}
/** 
 * @brief   CR-PCR solver: cr_forward_multiple + pcr_forward_single + cr_backward_multiple
 * @param   a_mpi (input) Lower off-diagonal coeff., which is assigned to local private pointer a
 * @param   b_mpi (input) Diagonal coeff., which is assigned to local private pointer b
 * @param   c_mpi (input) Upper off-diagonal coeff.,, which is assigned to local private pointer c
 * @param   r_mpi (input) RHS vector, which is assigned to local private pointer r
 * @param   x_mpi (output) Solution vector, which is assigned to local private pointer x
*/
void tdma_parallel :: cr_pcr_solver(double *a_mpi, double *b_mpi, double *c_mpi, double *r_mpi, double *x_mpi)
{
    a = a_mpi;
    b = b_mpi;
    c = c_mpi;
    r = r_mpi;
    x = x_mpi;

    cr_forward_multiple_row();
    pcr_forward_single_row();     // Including 2x2 solver
    cr_backward_multiple_row();
}
/** 
 * @brief   CR solver: cr_forward_multiple + cr_forward_single + cr_backward_single + cr_backward_multiple
 * @param   a_mpi (input) Lower off-diagonal coeff., which is assigned to local private pointer a
 * @param   b_mpi (input) Diagonal coeff., which is assigned to local private pointer b
 * @param   c_mpi (input) Upper off-diagonal coeff.,, which is assigned to local private pointer c
 * @param   r_mpi (input) RHS vector, which is assigned to local private pointer r
 * @param   x_mpi (output) Solution vector, which is assigned to local private pointer x
*/
void tdma_parallel :: cr_solver(double *a_mpi, double *b_mpi, double *c_mpi, double *r_mpi, double *x_mpi)
{
    a = a_mpi;
    b = b_mpi;
    c = c_mpi;
    r = r_mpi;
    x = x_mpi;

    cr_forward_multiple_row();
    cr_forward_single_row();
    cr_backward_single_row();     // Including 2x2 solver
    cr_backward_multiple_row();
}

/** 
 * @brief   Thomas-PCR solver: pThomas_forward_multiple + pcr_forward_double + pThomas_backward_multiple
 * @param   a_mpi (input) Lower off-diagonal coeff., which is assigned to local private pointer a
 * @param   b_mpi (input) Diagonal coeff., which is assigned to local private pointer b
 * @param   c_mpi (input) Upper off-diagonal coeff.,, which is assigned to local private pointer c
 * @param   r_mpi (input) RHS vector, which is assigned to local private pointer r
 * @param   x_mpi (output) Solution vector, which is assigned to local private pointer x
*/
void tdma_parallel :: Thomas_pcr_solver(double *a_mpi, double *b_mpi, double *c_mpi, double *r_mpi, double *x_mpi)
{
    a = a_mpi;
    b = b_mpi;
    c = c_mpi;
    r = r_mpi;
    x = x_mpi;

    pThomas_forward_multiple_row();
    pcr_double_row_substitution();
}

/** 
 * @brief   Forward elimination of CR until a single row per MPI process remains.
 * @details After a single row per MPI process remains, PCR or CR between a single row is performed.
*/
void tdma_parallel :: cr_forward_multiple_row()
{
    int i, l;
    int nlevel;
    int ip, in, start, dist_row, dist2_row;
    double alpha, gamma;
    double sbuf[4], rbuf[4];

    MPI_Status status, status1;
    MPI_Request request[2];

    /// Variable nlevel is used to indicates when single row remains.
    nlevel    = log2(n_mpi);
    dist_row  = 1;
    dist2_row = 2;
    
    for(l=0;l<nlevel;l++) {
        start = dist2_row;
        /// Data exchange is performed using MPI send/recv for each succesive reduction
        if(myrank<nprocs-1) {
            MPI_Irecv(rbuf, 4, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, request);
        }
        if(myrank>0) {
            sbuf[0] = a[dist_row];
            sbuf[1] = b[dist_row];
            sbuf[2] = c[dist_row];
            sbuf[3] = r[dist_row];
            MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, request+1);
        }
        if(myrank<nprocs-1) {
            MPI_Wait(request, &status1);
            a[n_mpi+1] = rbuf[0];
            b[n_mpi+1] = rbuf[1];
            c[n_mpi+1] = rbuf[2];
            r[n_mpi+1] = rbuf[3];
        }

        /// Odd rows of remained rows are reduced to even rows of remained rows in each reduction step.
        /// Index in of global last row is out of range, but we treat it as a = c = r = 0 and b = 1 in main function.
        for(i=start;i<=n_mpi;i+=dist2_row) {
            ip = i - dist_row;
            in = min(i + dist_row, n_mpi + 1);
            alpha = -a[i] / b[ip];
            gamma = -c[i] / b[in];

            b[i] += (alpha * c[ip] + gamma * a[in]);
            a[i] = alpha * a[ip];
            c[i] = gamma * c[in];
            r[i] += (alpha * r[ip] + gamma * r[in]);

        }
        /// As reduction continues, the indices of required coefficients doubles.
        dist2_row *= 2;
        dist_row *= 2;
        
        if(myrank>0) {
            MPI_Wait(request+1, &status);
        }
    }
}

/** 
 * @brief   Backward substitution of CR after single-row solution per MPI process is obtained.
*/
void tdma_parallel :: cr_backward_multiple_row()
{
    int i, l;
    int nlevel;
    int ip, in, dist_row, dist2_row;

    MPI_Status status;
    MPI_Request request[2];

    nlevel    = log2(n_mpi);
    dist_row = n_mpi/2;

    /// Each rank requires a solution on last row of previous rank.
    if(myrank>0) {
        MPI_Irecv(x,       1, MPI_DOUBLE, myrank-1, 100, MPI_COMM_WORLD, request);
    }
    if(myrank<nprocs-1) {
        MPI_Isend(x+n_mpi, 1, MPI_DOUBLE, myrank+1, 100, MPI_COMM_WORLD, request+1);
    }
    if(myrank>0) {
        MPI_Wait(request, &status);
    }
    for(l=nlevel-1;l>=0;l--) {
        dist2_row = dist_row * 2;
        for(i=n_mpi-dist_row;i>=0;i-=dist2_row) {
            ip = i - dist_row;
            in = i + dist_row;
            x[i] = r[i]-c[i]*x[in]-a[i]*x[ip];
            x[i] = x[i]/b[i];
        }
        dist_row = dist_row / 2;
    }
    if(myrank<nprocs-1) {
        MPI_Wait(request+1, &status);
    }
}

/** 
 * @brief   Forward elimination of CR between a single row per MPI process.
 */
void tdma_parallel :: cr_forward_single_row()
{
    int i, l;
    int nlevel, nhprocs;
    int ip, in, dist_rank, dist2_rank;
    double alpha, gamma, det;
    double sbuf[4], rbuf0[4], rbuf1[4];

    MPI_Status status;
    MPI_Request request[4];

    nlevel      = log2(nprocs);
    nhprocs     = nprocs/2;

    dist_rank  = 1;
    dist2_rank = 2;

    /// Cyclic reduction continues until 2x2 matrix are made in rank of nprocs-1 and nprocs/2-1
    for(l=0;l<nlevel-1;l++) {

        /// Odd rows of remained rows are reduced to even rows of remained rows in each reduction step.
        /// Coefficients are updated for even rows only.

        if((myrank+1)%dist2_rank == 0) {
            if(myrank-dist_rank>=0) {
                MPI_Irecv(rbuf0, 4, MPI_DOUBLE, myrank-dist_rank, 400, MPI_COMM_WORLD, request+2);
            }
            if(myrank+dist_rank<nprocs) {
                MPI_Irecv(rbuf1, 4, MPI_DOUBLE, myrank+dist_rank, 401, MPI_COMM_WORLD, request+3);
            }
            if(myrank-dist_rank>=0) {
                MPI_Wait(request+2, &status);
                a[0] = rbuf0[0];
                b[0] = rbuf0[1];
                c[0] = rbuf0[2];
                r[0] = rbuf0[3];
            }
            if(myrank+dist_rank<nprocs) {
                MPI_Wait(request+3, &status);
                a[n_mpi+1] = rbuf1[0];
                b[n_mpi+1] = rbuf1[1];
                c[n_mpi+1] = rbuf1[2];
                r[n_mpi+1] = rbuf1[3];
            }

            i = n_mpi;
            ip = 0;
            in = i + 1;
            alpha = -a[i] / b[ip];
            gamma = -c[i] / b[in];

            b[i] += (alpha * c[ip] + gamma * a[in]);
            a[i]  = alpha * a[ip];
            c[i]  = gamma * c[in];
            r[i] += (alpha * r[ip] + gamma * r[in]);
        }
        else if((myrank+1)%dist2_rank == dist_rank) {

            sbuf[0] = a[n_mpi];
            sbuf[1] = b[n_mpi];
            sbuf[2] = c[n_mpi];
            sbuf[3] = r[n_mpi];

            if(myrank+dist_rank<nprocs) {
                MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank+dist_rank, 400, MPI_COMM_WORLD, request);
            }
            if(myrank-dist_rank>=0) {
                MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank-dist_rank, 401, MPI_COMM_WORLD, request+1);
            }
            if(myrank+dist_rank<nprocs) MPI_Wait(request, &status);
            if(myrank-dist_rank>=0) MPI_Wait(request+1, &status);
        }
        dist_rank  *= 2;
        dist2_rank *= 2;
    }

    /// Solving 2x2 matrix. Rank of nprocs-1 and nprocs/2-1 solves it simultaneously.

    if(myrank==nhprocs-1) {
        MPI_Irecv(rbuf1, 4, MPI_DOUBLE, myrank+nhprocs, 402, MPI_COMM_WORLD, request+2);

        sbuf[0] = a[n_mpi];
        sbuf[1] = b[n_mpi];
        sbuf[2] = c[n_mpi];
        sbuf[3] = r[n_mpi];
        MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank+nhprocs, 403, MPI_COMM_WORLD, request);

        MPI_Wait(request+2, &status);
        a[n_mpi+1] = rbuf1[0];
        b[n_mpi+1] = rbuf1[1];
        c[n_mpi+1] = rbuf1[2];
        r[n_mpi+1] = rbuf1[3];

        i = n_mpi;
        in = n_mpi+1;
        det = b[i]*b[in] - c[i]*a[in];
        x[i] = (r[i]*b[in] - r[in]*c[i])/det;
        x[in] = (r[in]*b[i] - r[i]*a[in])/det;
        MPI_Wait(request, &status);
    }
    else if(myrank==nprocs-1) {
        MPI_Irecv(rbuf0, 4, MPI_DOUBLE, myrank-nhprocs, 403, MPI_COMM_WORLD, request+3);

        sbuf[0] = a[n_mpi];
        sbuf[1] = b[n_mpi];
        sbuf[2] = c[n_mpi];
        sbuf[3] = r[n_mpi];
        MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank-nhprocs, 402, MPI_COMM_WORLD, request+1);

        MPI_Wait(request+3, &status);
        a[0] = rbuf0[0];
        b[0] = rbuf0[1];
        c[0] = rbuf0[2];
        r[0] = rbuf0[3];

        ip = 0;
        i = n_mpi;
        det = b[ip]*b[i] - c[ip]*a[i];
        x[ip] = (r[ip]*b[i] - r[i]*c[ip])/det;
        x[i] = (r[i]*b[ip] - r[ip]*a[i])/det;
        MPI_Wait(request+1, &status);
    }
}

/** 
 * @brief   Backward substitution of CR until every MPI process gets solution for its single row.
*/
void tdma_parallel :: cr_backward_single_row()
{
    int i, l;
    int nlevel, nhprocs;
    int ip, in, dist_rank, dist2_rank;

    MPI_Status status;
    MPI_Request request[4];

    nlevel      = log2(nprocs);
    nhprocs     = nprocs/2;
    dist_rank   = nhprocs/2;
    dist2_rank  = nhprocs;

    /// Back substitution continues until all ranks obtains a solution on last row.
    for(l=nlevel-2;l>=0;l--) {

        if((myrank+1)%dist2_rank == 0) {
            if(myrank+dist_rank<nprocs) {
                MPI_Isend(x+n_mpi, 1, MPI_DOUBLE, myrank+dist_rank, 500, MPI_COMM_WORLD, request);
            }
            if(myrank-dist_rank>=0) {
                MPI_Isend(x+n_mpi, 1, MPI_DOUBLE, myrank-dist_rank, 501, MPI_COMM_WORLD, request+1);
            }
            if(myrank+dist_rank<nprocs) MPI_Wait(request, &status);
            if(myrank-dist_rank>=0) MPI_Wait(request+1, &status);

        }
        /// Only Odd rows of each level calculate new solution using a couple of even rows.
        else if((myrank+1)%dist2_rank == (dist_rank)) {
            if(myrank-dist_rank>=0) {
                MPI_Irecv(x,         1, MPI_DOUBLE, myrank-dist_rank, 500, MPI_COMM_WORLD, request+2);
            }
            if(myrank+dist_rank<nprocs) {
                MPI_Irecv(x+n_mpi+1, 1, MPI_DOUBLE, myrank+dist_rank, 501, MPI_COMM_WORLD, request+3);
            }
            if(myrank-dist_rank>=0) MPI_Wait(request+2, &status);
            if(myrank+dist_rank<nprocs) MPI_Wait(request+3, &status);

            i=n_mpi;
            ip = 0;
            in = n_mpi+1;
            x[i] = r[i]-c[i]*x[in]-a[i]*x[ip];
            x[i] = x[i]/b[i];
        }
        dist_rank /= 2;
        dist2_rank /= 2;
    }

}

/** 
 * @brief   PCR between a single row per MPI process and 2x2 matrix solver between i and i+nprocs/2 rows. 
*/
void tdma_parallel :: pcr_forward_single_row()
{

    int i, l, nhprocs;
    int nlevel;
    int ip, in, dist_rank, dist2_rank;
    int myrank_level, nprocs_level;
    double alpha, gamma, det;
    double sbuf[4], rbuf0[4], rbuf1[4];

    MPI_Status status;
    MPI_Request request[4];

    nlevel      = log2(nprocs);
    nhprocs     = nprocs/2;
    dist_rank   = 1;
    dist2_rank  = 2;

    /// Parallel cyclic reduction continues until 2x2 matrix are made between a pair of rank, 
    /// (myrank, myrank+nhprocs).
    for(l=0;l<nlevel-1;l++) {

        /// Rank is newly calculated in each level to find communication pair.
        /// Nprocs is also newly calculated as myrank is changed.
        myrank_level = myrank / dist_rank;
        nprocs_level = nprocs / dist_rank;

        /// All rows exchange data for reduction and perform reduction successively.
        /// Coefficients are updated for every rows.
        sbuf[0] = a[n_mpi];
        sbuf[1] = b[n_mpi];
        sbuf[2] = c[n_mpi];
        sbuf[3] = r[n_mpi];

        if((myrank_level+1)%2 == 0) {
            if(myrank+dist_rank<nprocs) {
                MPI_Irecv(rbuf1, 4, MPI_DOUBLE, myrank+dist_rank, 202, MPI_COMM_WORLD, request);
                MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank+dist_rank, 203, MPI_COMM_WORLD, request+1);
            }
            if(myrank-dist_rank>=0) {
                MPI_Irecv(rbuf0, 4, MPI_DOUBLE, myrank-dist_rank, 200, MPI_COMM_WORLD, request+2);
                MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank-dist_rank, 201, MPI_COMM_WORLD, request+3);
            }
            if(myrank+dist_rank<nprocs) {
                MPI_Wait(request, &status);
                a[n_mpi+1] = rbuf1[0];
                b[n_mpi+1] = rbuf1[1];
                c[n_mpi+1] = rbuf1[2];
                r[n_mpi+1] = rbuf1[3];
                MPI_Wait(request+1, &status);
            }
            if(myrank-dist_rank>=0) {
                MPI_Wait(request+2, &status);
                a[0] = rbuf0[0];
                b[0] = rbuf0[1];
                c[0] = rbuf0[2];
                r[0] = rbuf0[3];
                MPI_Wait(request+3, &status);
            }
        }
        else if((myrank_level+1)%2 == 1) {
            if(myrank+dist_rank<nprocs) {
                MPI_Irecv(rbuf1, 4, MPI_DOUBLE, myrank+dist_rank, 201, MPI_COMM_WORLD, request);
                MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank+dist_rank, 200, MPI_COMM_WORLD, request+1);
            }
            if(myrank-dist_rank>=0) {
                MPI_Irecv(rbuf0, 4, MPI_DOUBLE, myrank-dist_rank, 203, MPI_COMM_WORLD, request+2);
                MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank-dist_rank, 202, MPI_COMM_WORLD, request+3);
            }
            if(myrank+dist_rank<nprocs) {
                MPI_Wait(request, &status);
                a[n_mpi+1] = rbuf1[0];
                b[n_mpi+1] = rbuf1[1];
                c[n_mpi+1] = rbuf1[2];
                r[n_mpi+1] = rbuf1[3];
                MPI_Wait(request+1, &status);
            }
            if(myrank-dist_rank>=0) {
                MPI_Wait(request+2, &status);
                a[0] = rbuf0[0];
                b[0] = rbuf0[1];
                c[0] = rbuf0[2];
                r[0] = rbuf0[3];
                MPI_Wait(request+3, &status);
            }
        }

        i = n_mpi;
        ip = 0;
        in = i + 1;
        if(myrank_level == 0) {
            alpha = 0.0;
        }
        else {
            alpha = -a[i] / b[ip];
        }
        if(myrank_level == nprocs_level-1) {
            gamma = 0.0;
        }
        else {
            gamma = -c[i] / b[in];
        }

        b[i] += (alpha * c[ip] + gamma * a[in]);
        a[i]  = alpha * a[ip];
        c[i]  = gamma * c[in];
        r[i] += (alpha * r[ip] + gamma * r[in]);

        dist_rank  *= 2;
        dist2_rank *= 2;
    }

    /// Solving 2x2 matrix. All pair of ranks, myrank and myrank+nhprocs, solves it simultaneously.
    sbuf[0] = a[n_mpi];
    sbuf[1] = b[n_mpi];
    sbuf[2] = c[n_mpi];
    sbuf[3] = r[n_mpi];
    if(myrank<nhprocs) {
        MPI_Irecv(rbuf1, 4, MPI_DOUBLE, myrank+nhprocs, 300, MPI_COMM_WORLD, request);
        MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank+nhprocs, 301, MPI_COMM_WORLD, request+1);

        MPI_Wait(request, &status);
        a[n_mpi+1] = rbuf1[0];
        b[n_mpi+1] = rbuf1[1];
        c[n_mpi+1] = rbuf1[2];
        r[n_mpi+1] = rbuf1[3];

        i = n_mpi;
        in = n_mpi+1;

        det = b[i]*b[in] - c[i]*a[in];
        x[i] = (r[i]*b[in] - r[in]*c[i])/det;
        x[in] = (r[in]*b[i] - r[i]*a[in])/det;
        MPI_Wait(request+1, &status);

    }
    else if(myrank>=nhprocs) {
        MPI_Irecv(rbuf0, 4, MPI_DOUBLE, myrank-nhprocs, 301, MPI_COMM_WORLD, request+2);
        MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank-nhprocs, 300, MPI_COMM_WORLD, request+3);

        MPI_Wait(request+2, &status);
        a[0] = rbuf0[0];
        b[0] = rbuf0[1];
        c[0] = rbuf0[2];
        r[0] = rbuf0[3];

        ip = 0;
        i = n_mpi;

        det = b[ip]*b[i] - c[ip]*a[i];
        x[ip] = (r[ip]*b[i] - r[i]*c[ip])/det;
        x[i] = (r[i]*b[ip] - r[ip]*a[i])/det;
        MPI_Wait(request+3, &status);
    }
}
/** 
 * @brief   First phase of hybrid Thomas and PCR algorithm
 * @detail  Forward and backward elimination to remain two equations of first and last rows for each MPI processes
*/
void tdma_parallel :: pThomas_forward_multiple_row()
{
    int i;
    double alpha, beta;

    for(i=3;i<=n_mpi;i++) {
        alpha = - a[i] / b[i-1];
        a[i]  = alpha * a[i-1];
        b[i] += alpha * c[i-1];
        r[i] += alpha * r[i-1];
    }
    for(i=n_mpi-2;i>=1;i--) {
        beta  = - c[i] / b[i+1];
        c[i]  = beta * c[i+1];
        r[i] += beta * r[i+1];
        if(i==1) {
            b[1] += beta * a[2];
        }
        else
        {
            a[i] += beta * a[i+1];
        }
    }
}

/** 
 * @brief   PCR solver for two equations per each MPI process
 * @detail  Forward CR to remain a single equation per each MPI process.
 *          PCR solver for single row is, then, executed.
 *          Substitution is also performed to obtain every solution.
*/
void tdma_parallel :: pcr_double_row_substitution()
{
    int i, ip, in;
    double alpha, gamma;
    double sbuf[4], rbuf[4];

    MPI_Status status, status1;
    MPI_Request request[2];

    /// Cyclic reduction until single row remains per MPI process.
    /// First row of next rank is sent to current rank at the row of n_mpi+1 for reduction.
    if(myrank<nprocs-1) {
        MPI_Irecv(rbuf, 4, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, request);
    }
    if(myrank>0) {
        sbuf[0] = a[1];
        sbuf[1] = b[1];
        sbuf[2] = c[1];
        sbuf[3] = r[1];
        MPI_Isend(sbuf, 4, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, request+1);
    }
    if(myrank<nprocs-1) {
        MPI_Wait(request, &status1);
        a[n_mpi+1] = rbuf[0];
        b[n_mpi+1] = rbuf[1];
        c[n_mpi+1] = rbuf[2];
        r[n_mpi+1] = rbuf[3];
    }

    /// Every first row are reduced to the last row (n_mpi) in each MPI rank.
    i = n_mpi;
    ip = 1;
    in = i + 1;
    alpha = -a[i] / b[ip];
    gamma = -c[i] / b[in];

    b[i] += (alpha * c[ip] + gamma * a[in]);
    a[i] = alpha * a[ip];
    c[i] = gamma * c[in];
    r[i] += (alpha * r[ip] + gamma * r[in]);
    
    if(myrank>0) {
        MPI_Wait(request+1, &status);
    }

    /// Solution of last row in each MPI rank is obtained in pcr_forward_single_row().
    pcr_forward_single_row();

    /// Solution of first row in each MPI rank.
    if(myrank>0) {
        MPI_Irecv(x,       1, MPI_DOUBLE, myrank-1, 100, MPI_COMM_WORLD, request);
    }
    if(myrank<nprocs-1) {
        MPI_Isend(x+n_mpi, 1, MPI_DOUBLE, myrank+1, 100, MPI_COMM_WORLD, request+1);
    }
    if(myrank>0) {
        MPI_Wait(request, &status);
    }
    i = 1;
    ip = 0;
    in = n_mpi;
    x[1] = r[1]-c[1]*x[n_mpi]-a[1]*x[0];
    x[1] = x[1]/b[1];

    if(myrank<nprocs-1) {
        MPI_Wait(request+1, &status);
    }
    /// Solution of other rows in each MPI rank.
    for(int i=2;i<n_mpi;i++) {
        x[i] = r[i]-c[i]*x[n_mpi]-a[i]*x[1];
        x[i] = x[i]/b[i];
    }
}

/** 
 * @brief   Solution check
 * @param   *a_ver Coefficients of a with original values
 * @param   *b_ver Coefficients of b with original values
 * @param   *c_ver Coefficients of c with original values
 * @param   *r_ver RHS vector with original values
 * @param   *x_sol Solution vector
*/
void tdma_parallel :: verify_solution(double *a_ver, double *b_ver, double *c_ver, double *r_ver, double *x_sol)
{
    int i;
    double *y_ver = new double[n_mpi+2];

    MPI_Status status;
    MPI_Request request[4];

    if(myrank>0) {
        MPI_Isend(x+1, 1, MPI_DOUBLE, myrank-1, 900, MPI_COMM_WORLD, request);
        MPI_Irecv(x,   1, MPI_DOUBLE, myrank-1, 901, MPI_COMM_WORLD, request+1);
    }
    if(myrank<nprocs-1) {
        MPI_Isend(x+n_mpi,   1, MPI_DOUBLE, myrank+1, 901, MPI_COMM_WORLD, request+2);
        MPI_Irecv(x+n_mpi+1, 1, MPI_DOUBLE, myrank+1, 900, MPI_COMM_WORLD, request+3);
    }

    if(myrank>0) {
        MPI_Wait(request,   &status);
        MPI_Wait(request+1, &status);
    }
    if(myrank<nprocs-1) {
        MPI_Wait(request+2, &status);
        MPI_Wait(request+3, &status);
    }
    
    for(i=1;i<=n_mpi;i++) {
        y_ver[i] = a_ver[i]*x_sol[i-1]+b_ver[i]*x_sol[i]+c_ver[i]*x_sol[i+1];
        printf("Verify solution1 : myrank = %3d, a=%12.6f, b=%12.6f, c=%12.6f, x=%12.6f, r[%3d]=%12.6f, y[%3d]=%12.6f\n",myrank,a_ver[i],b_ver[i],c_ver[i],x_sol[i],i+n_mpi*myrank,r_ver[i],i+n_mpi*myrank,y_ver[i]);
    }
    delete [] y_ver;
}