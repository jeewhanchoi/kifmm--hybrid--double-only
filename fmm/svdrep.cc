#include <iostream>
#include <cstring> 
#include <assert.h>
#include <math.h>
#include "mkl.h"
#include "svdrep.h"

using namespace std;

int construct(SVDRep *svd, real_t epsilon, real_t* K, int m, int n)
{
  int k = min(m, n);
  
  real_t* tU = (real_t *) reals_alloc__aligned (m * k);
  real_t* tS = (real_t *) reals_alloc__aligned (k);
  real_t* tVT = (real_t *) reals_alloc__aligned (k * n);
  reals_zero (m*k, tU); 
  reals_zero (k, tS); 
  reals_zero (k*n, tVT); 

  //SVD
  int INFO;
  char JOBU  = 'S';
  char JOBVT = 'S';
  
  int wssize = max(3*min(m,n)+max(m,n), 5*min(m,n));
  real_t* wsbuf = (real_t *) reals_alloc__aligned (wssize);
  reals_zero (wssize, wsbuf);
  _GESVD(&JOBU, &JOBVT, &m, &n, K, &m, tS, tU, &m, tVT, &k, wsbuf, &wssize, &INFO);  assert(INFO==0);
  reals_free (wsbuf);
  
  //cutoff
  real_t cutoff = epsilon*tS[0];
  int cnt=0;
  while(cnt< k)
    if(fabs(tS[cnt]) >= cutoff)
      cnt++;
    else
      break;
  
  svd->matU = (real_t *) reals_alloc__aligned (m * cnt);
  svd->matS = (real_t *) reals_alloc__aligned (cnt);
  svd->matVT = (real_t *) reals_alloc__aligned (cnt * n);

  for(int i=0; i<m; i++) {
    for(int j=0; j<cnt; j++) {
      svd->matU[i+j*m] = tU[i+j*m];
    }
  }
  for(int i=0; i<cnt; i++) {
    svd->matS[i] = tS[i];
  }

  
  for(int i=0; i<cnt; i++) {
    for(int j=0; j<n; j++) {
      svd->matVT[i+j*cnt] = tVT[i+j*k];
    }
  }
  
  svd->m = m;
  svd->n = n;
  svd->r = cnt;
 
  return 0;
}
