/**
 * @brief       Parallel TDMA test subroutine
 * @author      Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
 * @date        15 January 2019
 * @version     0.1
 * @par         Copyright
                Copyright (c) 2019 by Ji-Hoon Kang. All rights reserved.
 * @par         License     
                This project is release under the terms of the MIT License (see LICENSE in )
*/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include "tdma_parallel.h"

using namespace std;

const int n = 64;

int main(int argc, char *argv[])
{
    int i, l, k;
    int nprocs, myrank;
    int n_mpi;

    tdma_parallel tdma;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    n_mpi = n / nprocs;

    double *a_mpi = new double[n_mpi+2];
    double *b_mpi = new double[n_mpi+2];
    double *c_mpi = new double[n_mpi+2];
    double *x_mpi = new double[n_mpi+2];
    double *r_mpi = new double[n_mpi+2];

    double *a_ver = new double[n_mpi+2];
    double *b_ver = new double[n_mpi+2];
    double *c_ver = new double[n_mpi+2];
    double *r_ver = new double[n_mpi+2];

    for(i=0;i<n_mpi+2;i++) {
        a_mpi[i] = 0.;
        b_mpi[i] = 1.;
        c_mpi[i] = 0.;
        r_mpi[i] = 0.;
        x_mpi[i] = 0.;
    }

    for(i=1;i<n_mpi+1;i++) {
        a_mpi[i] = 1.0; +0.01*(i+myrank*n_mpi);
        c_mpi[i] = 1.0; +0.02*(i+myrank*n_mpi);
        b_mpi[i] = -(a_mpi[i]+c_mpi[i])-0.1-0.02*(i+myrank*n_mpi)*(i+myrank*n_mpi);
        r_mpi[i] = (double)(i-1+myrank*n_mpi);
        a_ver[i] = a_mpi[i];
        b_ver[i] = b_mpi[i];
        c_ver[i] = c_mpi[i];
        r_ver[i] = r_mpi[i];
    }

    tdma.setup(n, nprocs, myrank);
//    tdma.cr_pcr_solver(a_mpi,b_mpi,c_mpi,r_mpi,x_mpi);
    tdma.Thomas_pcr_solver(a_mpi,b_mpi,c_mpi,r_mpi,x_mpi);
    tdma.verify_solution(a_ver,b_ver,c_ver,r_ver,x_mpi);

    delete[] x_mpi;
    delete[] r_mpi;
    delete[] a_mpi;
    delete[] b_mpi;
    delete[] c_mpi;
    delete[] a_ver;
    delete[] b_ver;
    delete[] c_ver;

    if(MPI_Finalize()) {
        cout << "MPI _Finalize error" << endl;
        exit(0);
    }
    return 0;
}
