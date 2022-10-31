#pragma once

#include <vector>


#define CGTOL (1e-3f)
#define CGMAXITER (10000)


#define PRINT_X



int cucg(int N, int nz, int* I, int* J, float* val, float* b);
int cupcg(int N, int nz, int* I, int* J, float* val, float* rhs);
int cucggraph(int N, int nz, int* I, int* J, float* val, float* rhs);