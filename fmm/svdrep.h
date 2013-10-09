#ifndef _SVDREP_H_
#define _SVDREP_H_

#include "reals_aligned.h"


/** SVD - Singular Value Decomposition Rrepresentation of some input matrix
 * The SVD Decomposition of a matrix, A, can be written as U*S*V^T where S is
 * a diagonal matrix of singular values, U and V are orthogonal matrices.  This decompition
 * allows for fast solving a a system of linear equations.  Here, the Lapack Fortran
 * code for DGESVD is called.  See lapack.h for more information */

typedef struct 
{
  int m, n, r;
  /* The matrix _matU from the decomposition U*S*V^T */
  real_t* matU;
  /* The vector _matS from the decomposition U*S*V^T : S is a matrix of diagonal values only, so is represented as a vector */
  real_t* matS;
  /* The matrix _matVT from the decomposition U*S*V^T */
  real_t* matVT;
} SVDRep;	

/* Construct the SVD representation of a matrix M with cutoff epsilon.  The smallest singular values
* are dropped based on epsilon.  The Lapack routine DGESVD from lapack.h is called here as well */
int construct(SVDRep *svd, real_t epsilon, real_t* M, int m, int n);

#endif
