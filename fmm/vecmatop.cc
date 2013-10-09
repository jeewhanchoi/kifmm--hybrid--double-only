#include <cstring> 
#include <stdio.h>
#include <assert.h>
#include "svdrep.h"
#include "mkl.h"
#include "vecmatop.h"

int daxpy(int n, real_t a, real_t* X, real_t* Y)
{
  int incx = 1;  int incy = 1;
  _AXPY(&n, &a, X, &incx, Y, &incy);
  return 0;
}
// ---------------------------------------------------------------------- 
int dgemv(int m, int n, real_t alpha, real_t* A, real_t* X, real_t beta, real_t* Y)
{
  char trans = 'N';
  int incx = 1;
  int incy = 1;
  _GEMV(&trans, &m, &n, &alpha, A, &m, X, &incx, &beta, Y, &incy);
  return 0;
}
// ---------------------------------------------------------------------- 
int tran(const real_t* M, real_t* R, int m, int n)
{
  for(int i=0; i<m; i++) {
	 for(int j=0; j<n; j++) {
		R[j+i*n] = M[i+j*m];
    }
  }
  return 0;
}
// ----------------------------------------------------------------------
int pinv(real_t* M, real_t epsilon, real_t* R, int m, int n )
{
  SVDRep *svd;
  svd = (SVDRep*) malloc (sizeof(SVDRep));
  assert (svd);
 
  /* Compute SVD */
  construct(svd, epsilon, M, m, n);

  double cutoff = epsilon * svd->matS[0];  //double cutoff = epsilon;
  for(int i=0; i<svd->r; i++) {
    if( svd->matS[i] >= cutoff) {
      svd->matS[i] = 1.0/(svd->matS[i]);
	 } else {
		assert(0);      //svd.S()(i) = 0.0;
	 }
  }
 
  real_t* UT = (real_t *) reals_alloc__aligned (svd->r * svd->m);
  assert (UT);
  real_t* V = (real_t *) reals_alloc__aligned (svd->n * svd->r);
  assert (V);

  tran(svd->matU, UT, svd->m, svd->r);

  tran(svd->matVT, V, svd->r, svd->n);
   
  for(int i=0; i<svd->n; i++) {
    for(int j=0; j<svd->r; j++) {
      V[i+j*svd->n] = V[i+j*svd->n] * svd->matS[j];
	  }
  }
  char transa = 'N';
  char transb = 'N';
  real_t alpha = 1.0;
  real_t beta = 0.0;
  int ms = svd->n;
  int ns = svd->m;
  int ks = svd->r;
  _GEMM(&transa, &transb, &ms, &ns, &ks, &alpha,
		  V, &ms, UT, &ks, 
		  &beta, R, &ms);  
  
  return 0;
}


