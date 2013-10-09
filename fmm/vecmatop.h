#ifndef _VECMATOP_H_
#define _VECMATOP_H_

#include "reals_aligned.h"

//--------------------------------------------------
//y = a x + y
int daxpy(int n, real_t a, real_t* X, real_t* Y);

// y <= alpha A x + beta y
int dgemv(int m, int n, real_t alpha, real_t* A, real_t* X, real_t beta, real_t* Y);

// R <= tran(M)
int tran(const real_t* M, real_t* R, int m, int n);

// R <= pinv(M, epsilon)
int pinv(real_t* M, real_t epsilon, real_t* R, int m, int n);
//--------------------------------------------------

#endif
