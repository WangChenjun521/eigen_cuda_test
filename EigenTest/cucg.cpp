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
  * This sample implements a conjugate gradient solver on GPU
  * using CUBLAS and CUSPARSE
  *
  */

  // includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

#include "cucg.h"

const char* sSDKname = "conjugateGradient";


int verifyCuda(int argc, char** argv) {
    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char**)argv);

    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
        deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
}



int cucg(int N, int nz, int* I, int* J, float* val, float* b) {
#if 0
    for (int i = nz - 20; i < nz; i++)
        if (i >= 0)
            printf("%3d   %ef\n", i + 4, val[i]);
#endif

    auto t0_ = GetTickCount();
    printf("CUCG    A: %d x %d, CSC,  b: %d, nz: %d\n", N, N, N, nz);
    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    checkCudaErrors(cusparseStatus);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int* d_col, * d_row;
    float* d_val, * d_x, dot;
    float* d_r, * d_p, * d_Ax;
    checkCudaErrors(cudaMalloc((void**)&d_col, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_row, (N + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_val, nz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_x, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_r, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_p, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_Ax, N * sizeof(float)));


    float* x = (float*)malloc(sizeof(float) * N);
    memset(x, 0, sizeof(float) * N);


    printf("cuda memcpy, %d ms\n", GetTickCount() - t0_);
    cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, b, N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0;
    float alpham1 = -1.0;
    float beta = 0.0;
    float r0 = 0.;
    float f, g, nf, r1;

    printf("cusparseScsrmv, %d ms\n", GetTickCount() - t0_);

    cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);


    
    const float tol = CGTOL;
    const int max_iter = CGMAXITER;
    printf("tol: [ %.8f ], max iter: [ %d ]\n", tol, max_iter);

    auto t1_ = GetTickCount();


    int k = 1;
    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            g = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &g, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        f = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &f, d_p, 1, d_x, 1);
        nf = -f;
        cublasStatus = cublasSaxpy(cublasHandle, N, &nf, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        //cudaDeviceSynchronize();
        //printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    auto t2_ = GetTickCount();
    
    cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    float rsum, diff, err = 0.0;
    for (int i = 0; i < N; i++) {
        rsum = 0.0;
        for (int j = I[i]; j < I[i + 1]; j++)
            rsum += val[j] * x[J[j]];

        diff = fabs(rsum - b[i]);
        if (diff > err)
            err = diff;
    }

    int info = 0;
    if (k > max_iter)
        info = 2;

    printf("solve -> %d, iter: %d, tol: %ef, err: %ef,  [ %d ] ms\n\n", info, k, sqrt(r1), err, t2_ - t1_);

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);


#ifdef PRINT_X
    printf("x:\n");
    for (int i = 0; i < N; i++)
        printf("%8.4f ", x[i]);
    printf("\n\n");
#endif
    printf("\n\n");

    free(x);
    return info;
}




