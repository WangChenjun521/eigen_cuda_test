/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a preconditioned conjugate gradient solver on
 * the GPU using CUBLAS and CUSPARSE.  Relative to the conjugateGradient
 * SDK example, this demonstrates the use of cusparseScsrilu0() for
 * computing the incompute-LU preconditioner and cusparseScsrsv_solve()
 * for solving triangular systems.  Specifically, the preconditioned
 * conjugate gradient method with an incomplete LU preconditioner is
 * used to solve the Laplacian operator in 2D on a uniform mesh.
 *
 * Note that the code in this example and the specific matrices used here
 * were chosen to demonstrate the use of the CUSPARSE library as simply
 * and as clearly as possible.  This is not optimized code and the input
 * matrices have been chosen for simplicity rather than performance.
 * These should not be used either as a performance guide or for
 * benchmarking purposes.
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA Runtime
#include <cuda_runtime.h>

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper for CUDA error checking

#include "cucg.h"


/* genLaplace: Generate a matrix representing a second order, regular, Laplacian operator on a 2d domain in Compressed Sparse Row format*/
void genLaplace(int *row_ptr, int *col_ind, float *val, int M, int N, int nz, float *b)
{
    assert(M==N);
    int n=(int)sqrt((double)N);
    assert(n*n==N);
    printf("laplace dimension = %d\n", n);
    int idx = 0;

    // loop over degrees of freedom
    for (int i=0; i<N; i++)
    {
        int ix = i%n;
        int iy = i/n;

        row_ptr[i] = idx;

        // up
        if (iy > 0)
        {
            val[idx] = 1.0;
            col_ind[idx] = i-n;
            idx++;
        }
        else
        {
            b[i] -= 1.0;
        }

        // left
        if (ix > 0)
        {
            val[idx] = 1.0;
            col_ind[idx] = i-1;
            idx++;
        }
        else
        {
            b[i] -= 0.0;
        }

        // center
        val[idx] = -4.0;
        col_ind[idx]=i;
        idx++;

        //right
        if (ix  < n-1)
        {
            val[idx] = 1.0;
            col_ind[idx] = i+1;
            idx++;
        }
        else
        {
            b[i] -= 0.0;
        }

        //down
        if (iy  < n-1)
        {
            val[idx] = 1.0;
            col_ind[idx] = i+n;
            idx++;
        }
        else
        {
            b[i] -= 0.0;
        }

    }
    row_ptr[N] = idx;
}

/* Solve Ax=b using the conjugate gradient method a) without any preconditioning, b) using an Incomplete Cholesky preconditioner and c) using an ILU0 preconditioner. */
int cupcg(int N, int nz, int* I, int* J, float* val, float* b)
{
    printf("CUPCG    A: %d x %d, CSR,  b: %d, nz: %d\n", N, N, N, nz);

    const int max_iter = CGMAXITER;
    const float tol = CGTOL;

    int k;
    int *d_col, *d_row;  
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_zm1, *d_zm2, *d_rm2;
    float *d_r, *d_p, *d_omega, *d_y;
    float *d_valsILU0;
    float *valsILU0;
    float rsum, diff, err = 0.0;
    float dot, numerator, denominator, nalpha;
    const float floatone = 1.0;
    const float floatzero = 0.0;
    int nErrors = 0;


    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    checkCudaErrors(cublasStatus);

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    checkCudaErrors(cusparseStatus);

    /* Description of the A matrix*/
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    checkCudaErrors(cusparseStatus);

    /* Define the properties of the matrix */
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    /* Allocate required memory */
    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_y, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, N*sizeof(float)));


    float* x = (float*)malloc(sizeof(float) * N);
    memset(x, 0, sizeof(float) * N);

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, b, N*sizeof(float), cudaMemcpyHostToDevice);

#if 0

    /* Conjugate gradient without preconditioning.
       ------------------------------------------
       Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */

    auto t0_ = clock();

    printf("Convergence of conjugate gradient without preconditioning: \n");
    k = 0;
    r0 = 0;
    cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    while (r1 > tol*tol && k <= max_iter)
    {
        k++;

        if (k == 1)
        {
            cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }
        else
        {
            beta = r1/r0;
            cublasSscal(cublasHandle, N, &beta, d_p, 1);
            cublasSaxpy(cublasHandle, N, &floatone, d_r, 1, d_p, 1) ;
        }

        cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega);
        cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
        alpha = r1/dot;
        cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
        nalpha = -alpha;
        cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
        r0 = r1;
        cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    }

    printf("%d ms\n", clock() - t0_);

    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    /* check result */
    err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - b[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr1 = err;

    if (0)
    {
        // output result in matlab-style array
        int n=(int)sqrt((double)N);
        printf("a = [  ");

        for (int iy=0; iy<n; iy++)
        {
            for (int ix=0; ix<n; ix++)
            {
                printf(" %f ", x[iy*n+ix]);
            }

            if (iy == n-1)
            {
                printf(" ]");
            }

            printf("\n");
        }
    }
#endif


    /* Preconditioned Conjugate Gradient using ILU.
       --------------------------------------------
       Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1  */        

    int nzILU0 = 2*N-1;
    valsILU0 = (float *) malloc(nz*sizeof(float));

    checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm1, (N)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_zm2, (N)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_rm2, (N)*sizeof(float)));

    /* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);
    checkCudaErrors(cusparseStatus);

    /* Perform the analysis for the Non-Transpose case */
    cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             N, nz, descr, d_val, d_row, d_col, infoA);
    checkCudaErrors(cusparseStatus);  
    

    /* Copy A data to ILU0 vals as input*/
    cudaMemcpy(d_valsILU0, d_val, nz*sizeof(float), cudaMemcpyDeviceToDevice);
    checkCudaErrors(cusparseStatus);

    /* Create info objects for the ILU0 preconditioner */
    cusparseSolveAnalysisInfo_t info_u;
    cusparseCreateSolveAnalysisInfo(&info_u);

    cusparseMatDescr_t descrL = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrL);
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseMatDescr_t descrU = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrU);
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);

    /* reset the initial guess of the solution to zero */
    memset(x, 0.0, sizeof(float) * N);

    checkCudaErrors(cudaMemcpy(d_r, b, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));


    auto t2_ = clock();
    /* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
    cusparseStatus = cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_row, d_col, info_u);
    printf("analyze:  %d ms\n", clock() - t2_);
    cusparseStatus = cusparseScsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descr, d_valsILU0, d_row, d_col, infoA);
    printf("factorize as ILU:  %d ms\n", clock() - t2_);

    k = 0;
    cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    while (r1 > tol*tol && k <= max_iter)
    {
        // Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
        cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrL, d_valsILU0, d_row, d_col, infoA, d_r, d_y);
        // Back Substitution
        cusparseStatus = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrU, d_valsILU0, d_row, d_col, info_u, d_y, d_zm1);

        k++;

        if (k == 1)
        {
            cublasScopy(cublasHandle, N, d_zm1, 1, d_p, 1);
        }
        else
        {
            cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);
            cublasSdot(cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator);
            beta = numerator/denominator;
            cublasSscal(cublasHandle, N, &beta, d_p, 1);
            cublasSaxpy(cublasHandle, N, &floatone, d_zm1, 1, d_p, 1) ;
        }

        cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nzILU0, &floatone, descrU, d_val, d_row, d_col, d_p, &floatzero, d_omega);
        cublasSdot(cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);
        cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &denominator);
        alpha = numerator / denominator;
        cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
        cublasScopy(cublasHandle, N, d_r, 1, d_rm2, 1);
        cublasScopy(cublasHandle, N, d_zm1, 1, d_zm2, 1);
        nalpha = -alpha;
        cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
        cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    }
    auto t2 = clock() - t2_;

    int info = 0;
    if (k > max_iter)
        info = 2;


    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    /* check result */
    err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - b[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    printf("solve -> %d, iter: %d, tol: %ef, err: %ef,  [ %d ] ms\n\n", info, k, sqrt(r1), err, t2);
#ifdef PRINT_X
    printf("x:\n");
    for (int i = 0; i < N; i++) {
        printf("%8.4f ", x[i]);
    }
#endif



    /* Destroy parameters */
    cusparseDestroySolveAnalysisInfo(infoA);
    cusparseDestroySolveAnalysisInfo(info_u);

    /* Destroy contexts */
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    /* Free device memory */
    free(x);
    free(valsILU0);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_omega);
    cudaFree(d_valsILU0);
    cudaFree(d_zm1);
    cudaFree(d_zm2);
    cudaFree(d_rm2);

    printf("\n\n");

    return info;
}

