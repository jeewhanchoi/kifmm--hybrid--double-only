#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "trans.h"
#include "node.h"
#include "util.h"
#include "reals_aligned.h"
#include "vecmatop.h"
#include "evaluate.h"
#include "evaluate--basic.h"

int
trans_setup (Pos *SP, Pos *RP)
{
  int np = getenv__accuracy();

  compute_sampos (np, 1.0, &SP[UE]);
  compute_sampos (np+2, 3.0, &SP[UC]);
  compute_sampos (np, 3.0, &SP[DE]);
  compute_sampos (np, 1.0, &SP[DC]);

  compute_regpos (np, 1.0, RP);

  return 0;
}

/* ------------------------------------------------------------------------
 */

int 
compute_sampos(int np, real_t R, Pos *SP)
{
  int i;
  SP->n = np*np*np - (np-2)*(np-2)*(np-2);

  /*Allocate memory for x, y and z in SP */
  SP->x = (real_t *) reals_alloc__aligned (SP->n);
  SP->y = (real_t *) reals_alloc__aligned (SP->n);
  SP->z = (real_t *) reals_alloc__aligned (SP->n);

  real_t step = 2.0/(np-1);
  real_t init = -1.0;
  int cnt = 0;
  for(int i=0; i<np; i++)
   for(int j=0; j<np; j++)
    for(int k=0; k<np; k++) {
      if(i==0 || i==np-1 || j==0 || j==np-1 || k==0 || k==np-1) {
       real_t x = init + i*step;
       real_t y = init + j*step;
       real_t z = init + k*step;
       SP->x[cnt] = R*x;
       SP->y[cnt] = R*y;
       SP->z[cnt] = R*z;
       cnt++;
      }   
    }   

  return 0;
}

/* ------------------------------------------------------------------------
 */
int 
compute_regpos(int np, real_t R, Pos *RP)
{
  RP->n = 2*np*2*np*2*np;
  RP->x = reals_alloc__aligned(RP->n);
  RP->y = reals_alloc__aligned(RP->n);
  RP->z = reals_alloc__aligned(RP->n);
  real_t step = 2.0/(np-1);
  int cnt = 0;
  for (int k=0; k<2*np; k++)
	  for (int j=0; j<2*np; j++)
		  for (int i=0; i<2*np; i++) {
		    int gi = (i<np) ? i : i-2*np;
		    int gj = (j<np) ? j : j-2*np;
		    int gk = (k<np) ? k : k-2*np;
		    RP->x[cnt] = R * gi*step;
		    RP->y[cnt] = R * gj*step;
		    RP->z[cnt] = R * gk*step;
		    cnt ++;
		  }
  return 0;
}

/* ------------------------------------------------------------------------
 */

int
compute_localpos (Point3 center, real_t radius, Node *t, Pos *SP)
{
  int i; 
  for (i = 0; i < t->num_pts; i++) {
    t->x[i] = center[0] + radius * SP->x[i];
    t->y[i] = center[1] + radius * SP->y[i];
    t->z[i] = center[2] + radius * SP->z[i];
  }
  
  return 0;
}

/* ------------------------------------------------------------------------
*/

int 
eff_data_size(int tp)
{
  int np = getenv__accuracy();
  int effNum = (2*np+2)*(2*np)*(2*np);
  if(tp==UE || tp==DE)
	 return effNum;
  else
	 return effNum;
}

/* ------------------------------------------------------------------------
*/

real_t alt()
{
  int np = getenv__accuracy();
  return pow(0.1, np+1);
}

/* ------------------------------------------------------------------------
*/

int pln_size(int tp, Pos *SP)
{
  int srcDOF = 1;
  int trgDOF = 1;

  if (tp==UE || tp==DE)
	 return SP[tp].n * srcDOF;
  else
	 return SP[tp].n * trgDOF;
}

/* ------------------------------------------------------------------------
*/

int 
compute_UC2UE_mat (Trans_matrix *TM, Pos *SP)
{
  real_t R = 1;
  real_t *ud2c;

  int r = pln_size (UC, SP);
  int c = pln_size (UE, SP);
  ud2c = (real_t *) reals_alloc__aligned (r * c);
  
  real_t chkPosX[SP[UC].n];	 reals_zero(SP[UC].n, chkPosX);	 daxpy(SP[UC].n, R, SP[UC].x, chkPosX) ; //scale
  real_t chkPosY[SP[UC].n];	 reals_zero(SP[UC].n, chkPosY);	 daxpy(SP[UC].n, R, SP[UC].y, chkPosY) ; //scale
  real_t chkPosZ[SP[UC].n];	 reals_zero(SP[UC].n, chkPosZ);	 daxpy(SP[UC].n, R, SP[UC].z, chkPosZ) ; //scale

  real_t denPosX[SP[UE].n];	 reals_zero(SP[UE].n, denPosX);	 daxpy(SP[UE].n, R, SP[UE].x, denPosX) ; //scale
  real_t denPosY[SP[UE].n];	 reals_zero(SP[UE].n, denPosY);	 daxpy(SP[UE].n, R, SP[UE].y, denPosY) ; //scale
  real_t denPosZ[SP[UE].n];	 reals_zero(SP[UE].n, denPosZ);	 daxpy(SP[UE].n, R, SP[UE].z, denPosZ) ; //scale
	
  kernel (SP[UE].n, SP[UC].n, denPosX, denPosY, denPosZ, chkPosX ,chkPosY, chkPosZ, ud2c);
  
  TM->UC2UE = (real_t *) reals_alloc__aligned (c * r);
  TM->m = c;
  TM->n = r;
	pinv(ud2c, alt(), TM->UC2UE, r, c);
 
  return 0;
}

/* ------------------------------------------------------------------------
*/

int 
compute_UE2UC_mat (Index3 idx, Trans_matrix *TM, Pos *SP)
{
  real_t R = 1;
  real_t *ue2uc;

  int r = pln_size (UC, SP);
  int c = pln_size (UE, SP);
  ue2uc = (real_t *) reals_alloc__aligned (r * c);
  
  real_t chkPosX[SP[UC].n];	 reals_zero(SP[UC].n, chkPosX);	 daxpy(SP[UC].n, 2.0*R, SP[UC].x, chkPosX) ; //scale
  real_t chkPosY[SP[UC].n];	 reals_zero(SP[UC].n, chkPosY);	 daxpy(SP[UC].n, 2.0*R, SP[UC].y, chkPosY) ; //scale
  real_t chkPosZ[SP[UC].n];	 reals_zero(SP[UC].n, chkPosZ);	 daxpy(SP[UC].n, 2.0*R, SP[UC].z, chkPosZ) ; //scale
  real_t denPosX[SP[UE].n];	 reals_zero(SP[UE].n, denPosX);	 daxpy(SP[UE].n, R, SP[UE].x, denPosX) ; //scale
  real_t denPosY[SP[UE].n];	 reals_zero(SP[UE].n, denPosY);	 daxpy(SP[UE].n, R, SP[UE].y, denPosY) ; //scale
  real_t denPosZ[SP[UE].n];	 reals_zero(SP[UE].n, denPosZ);	 daxpy(SP[UE].n, R, SP[UE].z, denPosZ) ; //scale
  
  // shift
  for (int j = 0; j < SP[UE].n; j++) {
    denPosX[j] = denPosX[j] + (2 * idx(0) - 1) * R;
    denPosY[j] = denPosY[j] + (2 * idx(1) - 1) * R;
    denPosZ[j] = denPosZ[j] + (2 * idx(2) - 1) * R;
  }
  	 
	kernel (SP[UE].n, SP[UC].n, denPosX, denPosY, denPosZ, chkPosX ,chkPosY, chkPosZ, ue2uc);
 
  TM->UE2UC[idx(2)+idx(1)*2+idx(0)*2*2] = ue2uc;
  
  return 0;
}

/* ------------------------------------------------------------------------
*/

int 
compute_DC2DE_mat (Trans_matrix *TM, Pos *SP)
{
  real_t R = 1;
  real_t *dd2c;

  int r = pln_size (DC, SP);
  int c = pln_size (DE, SP);
  dd2c = (real_t *) reals_alloc__aligned (r * c);
  
  real_t chkPosX[SP[DC].n];	 reals_zero(SP[DC].n, chkPosX);	 daxpy(SP[DC].n, R, SP[DC].x, chkPosX) ; //scale
  real_t chkPosY[SP[DC].n];	 reals_zero(SP[DC].n, chkPosY);	 daxpy(SP[DC].n, R, SP[DC].y, chkPosY) ; //scale
  real_t chkPosZ[SP[DC].n];	 reals_zero(SP[DC].n, chkPosZ);	 daxpy(SP[DC].n, R, SP[DC].z, chkPosZ) ; //scale

  real_t denPosX[SP[DE].n];	 reals_zero(SP[DE].n, denPosX);	 daxpy(SP[DE].n, R, SP[DE].x, denPosX) ; //scale
  real_t denPosY[SP[DE].n];	 reals_zero(SP[DE].n, denPosY);	 daxpy(SP[DE].n, R, SP[DE].y, denPosY) ; //scale
  real_t denPosZ[SP[DE].n];	 reals_zero(SP[DE].n, denPosZ);	 daxpy(SP[DE].n, R, SP[DE].z, denPosZ) ; //scale
	
  kernel (SP[DE].n, SP[DC].n, denPosX, denPosY, denPosZ, chkPosX ,chkPosY, chkPosZ, dd2c);
  TM->DC2DE = (real_t *) reals_alloc__aligned (c * r);
  TM->m = c;
  TM->n = r;
	pinv(dd2c, alt(), TM->DC2DE, r, c);
  
  return 0;
}
/* ------------------------------------------------------------------------
*/

int 
compute_DE2DC_mat (Index3 idx, Trans_matrix *TM, Pos *SP)
{
  real_t R = 1;
  real_t *de2dc;

  int r = pln_size (DC, SP);
  int c = pln_size (DE, SP);
  de2dc = (real_t *) reals_alloc__aligned (r * c);
  
  real_t denPosX[SP[DE].n];	 reals_zero(SP[DE].n, denPosX);	 daxpy(SP[DE].n, R, SP[DE].x, denPosX) ; //scale
  real_t denPosY[SP[DE].n];	 reals_zero(SP[DE].n, denPosY);	 daxpy(SP[DE].n, R, SP[DE].y, denPosY) ; //scale
  real_t denPosZ[SP[DE].n];	 reals_zero(SP[DE].n, denPosZ);	 daxpy(SP[DE].n, R, SP[DE].z, denPosZ) ; //scale
  real_t chkPosX[SP[DC].n];	 reals_zero(SP[DC].n, chkPosX);	 daxpy(SP[DC].n, 0.5*R, SP[DC].x, chkPosX) ; //scale
  real_t chkPosY[SP[DC].n];	 reals_zero(SP[DC].n, chkPosY);	 daxpy(SP[DC].n, 0.5*R, SP[DC].y, chkPosY) ; //scale
  real_t chkPosZ[SP[DC].n];	 reals_zero(SP[DC].n, chkPosZ);	 daxpy(SP[DC].n, 0.5*R, SP[DC].z, chkPosZ) ; //scale
  
  // shift
  for (int j = 0; j < SP[DC].n; j++) {
    chkPosX[j] = chkPosX[j] + (idx(0) - 0.5) * R;
    chkPosY[j] = chkPosY[j] + (idx(1) - 0.5) * R;
    chkPosZ[j] = chkPosZ[j] + (idx(2) - 0.5) * R;
  }
  	 
	kernel (SP[DE].n, SP[DC].n, denPosX, denPosY, denPosZ, chkPosX ,chkPosY, chkPosZ, de2dc);
 
  TM->DE2DC[idx(2)+idx(1)*2+idx(0)*2*2] = de2dc;
  
  return 0;
}

/* ------------------------------------------------------------------------
*/

int 
plnDen2EffDeninit(int l, real_t* pln_den, real_t* eff_den, FFT_PLAN& forplan, AllNodes *All_N)
{
  int srcDOF = 1;
  Pos* SP = All_N->SP;
  Pos* RP = All_N->RP;

  real_t* reg_den = reals_alloc__aligned(RP->n*srcDOF); 
  reals_zero(RP->n, reg_den);

	real_t* tmp_den = reals_alloc__aligned(srcDOF*SP[UE].n);	 
  reals_zero(SP[UE].n, tmp_den);

  real_t degVec[1]; degVec[0] = 1;
	real_t sclvec[srcDOF];	 
  for(int s=0; s<srcDOF; s++)		
    sclvec[s] = pow(2.0, l*degVec[s]);

	int cnt = 0;
	for(int i=0; i<SP[UE].n; i++)
	  for(int s=0; s<srcDOF; s++) {
		  tmp_den[cnt] = pln_den[cnt] * sclvec[s];
		  cnt++;
		}
	samDen2RegDen (tmp_den, reg_den);
  
  int np = getenv__accuracy();
  int nnn[3];  nnn[0] = 2*np;  nnn[1] = 2*np;  nnn[2] = 2*np;
  forplan = FFT_CREATE(3, nnn, srcDOF, reg_den, NULL, srcDOF, 1, (FFT_COMPLEX*)(eff_den), NULL, srcDOF, 1, FFTW_ESTIMATE);
  FFT_EXECUTE(forplan);
  
  return (0);
}

/* ------------------------------------------------------------------------
*/

int 
plnDen2EffDen(int l, real_t* pln_den, real_t* eff_den, real_t* reg_den, real_t* tmp_den, FFT_PLAN& forplan, AllNodes *All_N)
{
  int srcDOF = 1;
  Pos* SP = All_N->SP;
  Pos* RP = All_N->RP;

  real_t degVec[1]; degVec[0] = 1;
	real_t sclvec[srcDOF];	 
  for(int s=0; s<srcDOF; s++)		
    sclvec[s] = pow(2.0, l*degVec[s]);

	int cnt = 0;
	for(int i=0; i<SP[UE].n; i++)
	  for(int s=0; s<srcDOF; s++) {
		  tmp_den[cnt] = pln_den[cnt] * sclvec[s];
		  cnt++;
		}
	samDen2RegDen (tmp_den, reg_den);
  
  FFT_RE_EXECUTE(forplan, reg_den, (FFT_COMPLEX*)(eff_den));

  return (0);
}

/* ------------------------------------------------------------------------
*/
int 
samDen2RegDen(const real_t* sam_den, real_t* reg_den)
{
  int np = getenv__accuracy();
  int rgnum = 2*np;
  int srcDOF = 1;
  int cnt=0;
  //the order of iterating is the same as SampleGrid
  for(int i=0; i<np; i++)
	  for(int j=0; j<np; j++)
		  for(int k=0; k<np; k++) {
		    if(i==0 || i==np-1 || j==0 || j==np-1 || k==0 || k==np-1) {
			    //the position is fortran style
			    int rgoff = (k+np/2)*rgnum*rgnum + (j+np/2)*rgnum + (i+np/2);
			    for(int f=0; f<srcDOF; f++) {
				    reg_den[srcDOF*rgoff + f] = sam_den[srcDOF*cnt + f];
			    }
			    cnt++;
		    }
		  }
  
  return 0;
}

/* ------------------------------------------------------------------------
*/

int 
effVal2PlnValinit(int l, real_t* eff_val, real_t* pln_val, FFT_PLAN& invplan, AllNodes *All_N)
{
  int trgDOF = 1;
  int np = getenv__accuracy();
  Pos* RP = All_N->RP;
  
  int nnn[3];  nnn[0] = 2*np;  nnn[1] = 2*np;  nnn[2] = 2*np;
  real_t* reg_val = reals_alloc__aligned(RP->n*trgDOF); 
  invplan = IFFT_CREATE(3, nnn, trgDOF, (FFT_COMPLEX*)(eff_val), NULL, trgDOF, 1, reg_val, NULL, trgDOF, 1, FFTW_ESTIMATE);
  IFFT_EXECUTE(invplan);
  
  regVal2SamVal(reg_val, pln_val);

  return (0);
}

/* ------------------------------------------------------------------------
*/

int 
effVal2PlnVal(int l, real_t* eff_val, real_t* pln_val, real_t* reg_val, FFT_PLAN invplan, AllNodes *All_N)
{
  IFFT_RE_EXECUTE(invplan, (FFT_COMPLEX*)(eff_val), reg_val); 
  regVal2SamVal(reg_val, pln_val);

  return (0);
}

/* ------------------------------------------------------------------------
*/

int 
regVal2SamVal(const real_t* reg_val, real_t* sam_val)
{
  int np = getenv__accuracy();
  int rgnum = 2*np;
  int trgDOF = 1;
  int cnt=0;
  //the order of iterating is the same as SampleGrid
  for(int i=0; i<np; i++)
	  for(int j=0; j<np; j++)
		  for(int k=0; k<np; k++) {
		    if(i==0 || i==np-1 || j==0 || j==np-1 || k==0 || k==np-1) {
			    //the position is fortran style
			    int rgoff = (k+np/2)*rgnum*rgnum + (j+np/2)*rgnum + (i+np/2);
			    for(int f=0; f<trgDOF; f++) {
				    sam_val[trgDOF*cnt + f] += reg_val[trgDOF*rgoff + f];
			    }
			    cnt++;
		    }
		  }

  return 0;
}

/* ------------------------------------------------------------------------
*/
