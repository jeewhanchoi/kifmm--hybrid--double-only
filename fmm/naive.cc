#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "evaluate--basic.h"
#include "evaluate.h"
#include "reals_aligned.h"
#include "trans.h"
#include "vecmatop.h"
#include "vec3t.h"
#include "input.h"
#include "timing.h"
/* ------------------------------------------------------------------------
 */

const char *
get_implementation_name (void)
{
  return "naive";
}

/* ------------------------------------------------------------------------
 */

int
get_num_threads (void)
{
  int nthreads = 1;
  return nthreads;
}

/* ------------------------------------------------------------------------
 */
int
up_calc__cpu (FMMWrapper_t *F)
{
  int i, n, np; 
  Point3 c;
  real_t r;
  Node* tl_pos;
  AllNodes *All_N = F->AN;
  Pos* SP = All_N->SP;
  Trans_matrix* TM = All_N->TM;
  vector<NodeTree>& nodeVec = *All_N->N;
  
  tl_pos = (Node *) _mm_malloc(sizeof(Node), IDEAL_ALIGNMENT);
    
  tl_pos->x = (real_t *) reals_alloc__aligned (SP[UC].n);
  tl_pos->y = (real_t *) reals_alloc__aligned (SP[UC].n);
  tl_pos->z = (real_t *) reals_alloc__aligned (SP[UC].n);
  tl_pos->den_pot = (real_t *) reals_alloc__aligned (SP[UC].n);
  tl_pos->num_pts = SP[UC].n;

  /* Pre-compute UC2UE matrix */
  compute_UC2UE_mat (TM, SP);

  /* Pre-compute UE2UC matrices */
  TM->UE2UC = (real_t **) malloc (sizeof(real_t*) * 2*2*2);

	for(int a=0; a<2; a++) {
	  for(int b=0; b<2; b++) {
			for(int c=0; c<2; c++) {
        Index3 idx(a, b, c);
        compute_UE2UC_mat (idx, TM, SP);
      }
    }
  }


  /* Upward */
  for (i = nodeVec.size()-1; i >= 0; i--) {
    if (nodeVec[i].tag & LET_SRCNODE) {
      if (nodeVec[i].depth >= 0) {
        real_t *uden = &All_N->src_upw_equ_den[i*pln_size(UE, SP)];
        /* S2M */
        if (nodeVec[i].child == -1) {
          /* Upward check potential */
          c = center (i, nodeVec);
          r = radius (i, nodeVec);
          compute_localpos (c, r, tl_pos, &SP[UC]);
          reals_zero (SP[UC].n, tl_pos->den_pot);
 
          ulist__direct_evaluation (*tl_pos, All_N->Ns[i]);
        }
        /* M2M */
        else {
          reals_zero (SP[UC].n, tl_pos->den_pot);
			    for(int a=0; a<2; a++) {
				    for(int b=0; b<2; b++) {
				      for(int c=0; c<2; c++) {
					      Index3 idx(a,b,c);
					      int chi = child(i, idx, nodeVec);
					      if(nodeVec[chi].tag & LET_SRCNODE) {
                  real_t *den = &All_N->src_upw_equ_den[chi*pln_size(UE, SP)];
	                int srcDOF = 1;
	                real_t *tmpDen = (real_t *) reals_alloc__aligned (srcDOF*SP[UE].n);	 
                  reals_zero (SP[UE].n, tmpDen);
                  int l = nodeVec[chi].depth;
                  real_t degVec[1]; degVec[0] = 1;
	                real_t sclvec[srcDOF];	 
                  for(int s=0; s<srcDOF; s++)		
                    sclvec[s] = pow(2.0, l*degVec[s]);
	                int cnt = 0;
	                for(int k=0; k < SP[UE].n; k++)
		                for(int s=0; s < srcDOF; s++) {
		                  tmpDen[cnt] = den[cnt] * sclvec[s];
                      cnt++;
                    }
                  real_t *UE2UCii = TM->UE2UC[idx(2)+idx(1)*2+idx(0)*2*2];
	                dgemv(TM->n, TM->m, 1.0, UE2UCii, tmpDen, 1.0, tl_pos->den_pot);
					      }
              }
            }
          }
        }
          
        /* Upward equivalent density */
	      int srcDOF = 1;
	      real_t *tmpDen = (real_t *) reals_alloc__aligned (srcDOF*SP[UE].n);	 
        reals_zero (SP[UE].n, tmpDen);
	      dgemv(TM->m, TM->n, 1.0, TM->UC2UE, tl_pos->den_pot, 1.0, tmpDen);

	      //scale
        int l = nodeVec[i].depth;
        real_t degVec[1]; degVec[0] = 1;
	      real_t sclvec[srcDOF];	 
        for(int s=0; s<srcDOF; s++)		
          sclvec[s] = pow(2.0, -l*degVec[s]);
	      int cnt = 0;
	      for(int k=0; k < SP[UE].n; k++)
		      for(int s=0; s < srcDOF; s++) {
		        uden[cnt] = uden[cnt] + tmpDen[cnt] * sclvec[s];
		        cnt++;
		    }
      }
    }
  }
  return 0;
}

/* ------------------------------------------------------------------------
 */

int
ulist_calc__cpu (FMMWrapper_t *F)
{
  AllNodes *All_N = F->AN;
  if (All_N) {
    int i;
    vector<NodeTree>& nodeVec = *All_N->N;

    for (i = 0; i < nodeVec.size(); i++) {
      int k;
      for (k = 0; k < nodeVec[i].Unodes.size(); k++) {
	      int src = nodeVec[i].Unodes[k];
	      ulist__direct_evaluation (All_N->Nt[i], All_N->Ns[src]);
      }
    }
  }
  return 0;
}

int 
ulist__direct_evaluation (Node trg, Node src) 
{
  int i, j;
  real_t x, y, z, r2, r;
  
  for (i = 0; i < trg.num_pts; i++) {
    real_t tx = trg.x[i];
    real_t ty = trg.y[i];
    real_t tz = trg.z[i];
    real_t td = 0.0;
    for (j = 0; j < src.num_pts; j++) {
      x = tx - src.x[j];
      y = ty - src.y[j];
      z = tz - src.z[j];
      r2 = x*x + y*y + z*z;
#if !defined (USE_FLOAT)
      r = sqrt(r2);
#else 
      r = sqrtf(r2);
#endif
      if (r != 0)
        td += src.den_pot[j] / r;
    }
    trg.den_pot[i]  += OOFP_R * td;
  }

  return 0;
}

int 
kernel (int sn, int tn, real_t* x1, real_t* x2, real_t* x3, real_t* y1, real_t* y2, real_t* y3,
real_t* mat) 
{
  int i, j;
  real_t x, y, z, r2, r;
 
  for (i = 0; i < sn; i++) {
    real_t sx = x1[i];
    real_t sy = x2[i];
    real_t sz = x3[i];
    real_t td = 0.0;
    for (j = 0; j < tn; j++) {
      x = sx - y1[j];
      y = sy - y2[j];
      z = sz - y3[j];
      r2 = x*x + y*y + z*z;
#if !defined (USE_FLOAT)
      r = sqrt(r2);
#else 
      r = sqrtf(r2);
#endif
      if (r != 0)
        mat[j+i*tn]  = OOFP_R / r;
    }
  }
  return 0;
}

/* ------------------------------------------------------------------------
 */
int
compute_fft_src (AllNodes *All_N, FFT_PLAN& forplan)
{
  int i = 0;
  int eff_size;
  real_t* eden;
  real_t* uden;
  real_t* rden;
  real_t* tden;
  vector<NodeTree>& nodeVec = *All_N->N;
  eff_size = eff_data_size (UE);
  Pos* SP = All_N->SP;
  Pos* RP = All_N->RP;
  rden = reals_alloc__aligned (RP->n);
  tden = reals_alloc__aligned (SP[UE].n);

  if (nodeVec[i].tag & LET_SRCNODE) {
    eden = &All_N->eff_den[i * eff_size];
    uden = &All_N->src_upw_equ_den[i*pln_size(UE, SP)];
		plnDen2EffDeninit(nodeVec[i].depth, uden, eden, forplan, All_N);			 //2. transform from upeDen to effDen
	}

  for(i = 1; i<nodeVec.size(); i++) {
    if (nodeVec[i].tag & LET_SRCNODE) {
      eden = &All_N->eff_den[i * eff_size];
      uden = &All_N->src_upw_equ_den[i*pln_size(UE, SP)];
		  plnDen2EffDen(nodeVec[i].depth, uden, eden, rden, tden, forplan, All_N);			 //2. transform from upeDen to effDen
	  }
  }

  return 0;
}

/* ------------------------------------------------------------------------
 */
int
compute_fft_trans (AllNodes *All_N, FFT_PLAN& forplan)
{
  int idx;
  int srcDOF;
  int trgDOF;
  int effNum =  eff_data_size (UE);
  real_t R = 1; // TODO: fix needed for vector kernel
  Pos* RP = All_N->RP;
  Trans_matrix* TM = All_N->TM;

  /* Pre-compute UE2UC matrices */
  TM->UE2DC = (real_t **) malloc (sizeof(real_t*) * 7*7*7);
	  
  for (int i1=-3; i1<=3; i1++)
    for (int i2=-3; i2<=3; i2++)
	    for (int i3=-3; i3<=3; i3++)
	      if (abs(i1)>1 || abs(i2)>1 || abs(i3)>1 ) {
	        // compute and copy translation operator
          idx = (i1+3) + (i2+3)*7 + (i3+3)*7*7;
	        srcDOF = 1;
	        trgDOF = 1;
	        real_t denPosX[1];	
	        real_t denPosY[1];	
	        real_t denPosZ[1];	
	        denPosX[0] = (real_t)(i1)*2.0*R; //shift
	        denPosY[0] = (real_t)(i2)*2.0*R; //shift
	        denPosZ[0] = (real_t)(i3)*2.0*R; //shift

          real_t chkPosX[RP->n];	 reals_zero(RP->n, chkPosX);	 daxpy(RP->n, R, RP->x, chkPosX) ; //scale
          real_t chkPosY[RP->n];	 reals_zero(RP->n, chkPosY);	 daxpy(RP->n, R, RP->y, chkPosY) ; //scale
          real_t chkPosZ[RP->n];	 reals_zero(RP->n, chkPosZ);	 daxpy(RP->n, R, RP->z, chkPosZ) ; //scale
	        real_t* tt = reals_alloc__aligned (RP->n*trgDOF*srcDOF);
	        kernel(1, RP->n, denPosX, denPosY, denPosZ, chkPosX, chkPosY, chkPosZ, tt);
	        
          // move data to tmp
	        real_t* tmp = reals_alloc__aligned (RP->n*trgDOF*srcDOF);
	        for(int k=0; k<RP->n;k++) {
			      tmp[k] = tt[k];
	        }
	        real_t *UpwEqu2DwnChkii = reals_alloc__aligned (trgDOF*srcDOF * effNum); 
	        FFT_RE_EXECUTE(forplan, tmp, (FFT_COMPLEX*)(UpwEqu2DwnChkii));   //TODO: When srcDOF = trgDOF = 1 same plan as prev FFT can be used. Otherwise new plan should be created.
          TM->UE2DC[idx] = UpwEqu2DwnChkii;
	      } 
  return 0;  
}

/* ------------------------------------------------------------------------
 */

int
compute_ifft_trg (AllNodes *All_N)
{
  int i = 0;
  int invset = 0;
  int eff_size;
  real_t *eval;
  real_t *tval;
  real_t *rval;
  FFT_PLAN invplan;
  vector<NodeTree>& nodeVec = *All_N->N;
  Pos* SP = All_N->SP;
  Pos* RP = All_N->RP;

  eff_size = eff_data_size (DC);
  rval = reals_alloc__aligned (RP->n);

  while(invset == 0) {
	  if( nodeVec[i].tag & LET_TRGNODE && nodeVec[i].Vnodes.size()>0) { 
      tval = &All_N->trg_dwn_chk_val[i*pln_size(DC, SP)];
      eval = &All_N->eff_val[i * eff_size];
  		real_t nrmfc = 1.0/real_t(RP->n);
    	for (int k=0; k<eff_size; k++)
      	eval[k] *= nrmfc;
		  effVal2PlnValinit(nodeVec[i].depth, eval, tval, invplan, All_N);			 //1. transform from effVal to dncVal
      invset = 1;
	  }
	  i++;
  }

  for(int j=i; j < nodeVec.size(); j++) {
	  if( nodeVec[j].tag & LET_TRGNODE && nodeVec[j].Vnodes.size()>0) { 
      tval = &All_N->trg_dwn_chk_val[j*pln_size(DC, SP)];
      eval = &All_N->eff_val[j * eff_size];
  		real_t nrmfc = 1.0/real_t(RP->n);
    	for (int k=0; k<eff_size; k++)
      	eval[k] *= nrmfc;
		  effVal2PlnVal(nodeVec[i].depth, eval, tval, rval, invplan, All_N);			 //1. transform from effVal to dncVal
	  }
  }
  FFT_DESTROY (invplan);
  
  return 0;  
}

/* ------------------------------------------------------------------------
 */

int
vlist_calc__cpu (FMMWrapper_t* F)
{
  AllNodes *All_N = F->AN;
  if (All_N) {
    
    int i, j, id;
    int eff_src_size;
    int eff_trg_size;
    int effNum;
    int dim = 3;
    int srcDOF = 1;
    int trgDOF = 1;
    real_t *eden;
    real_t *eval;
    real_t *UpwEqu2DwnChkii;
    vector<NodeTree>& nodeVec = *All_N->N;
    Pos* SP = All_N->SP;
    Pos* RP = All_N->RP;
    Trans_matrix* TM = All_N->TM;

    FFT_PLAN forplan;
    compute_fft_src (All_N, forplan);
    compute_fft_trans (All_N, forplan);
//    FFT_DESTROY(forplan);
    
    eff_trg_size = eff_data_size (DC);
    eff_src_size = eff_data_size (UE);
    effNum = eff_src_size;
    for (i = 0; i < nodeVec.size(); i++) {
	    if( nodeVec[i].tag & LET_TRGNODE && nodeVec[i].Vnodes.size()>0) { 
        eval = &All_N->eff_val[i * eff_trg_size];
		    Point3 gNodeIdxCtr(center(i, nodeVec));
		    real_t D = 2.0 * radius (i, nodeVec);
        for (j = 0; j < nodeVec[i].Vnodes.size(); j++) {
	        int src = nodeVec[i].Vnodes[j];
		      Point3 viCtr(center (src, nodeVec));
		      Index3 idx;
		      for(int d=0; d<dim; d++) {
			      idx(d) = int (round( (viCtr[d]-gNodeIdxCtr[d])/D ));
		      }
          eden = &All_N->eff_den[src * eff_src_size];
  		    //fft mult
          id = (idx(0)+3) + (idx(1)+3)*7 + (idx(2)+3)*7*7;
          UpwEqu2DwnChkii = TM->UE2DC[id];
  		    real_t nrmfc = 1.0/real_t(RP->n);
  		    FFT_COMPLEX* matPtr  = (FFT_COMPLEX*)(UpwEqu2DwnChkii);
  		    FFT_COMPLEX* denPtr = (FFT_COMPLEX*)(eden);
  		    FFT_COMPLEX* chkPtr   = (FFT_COMPLEX*)(eval);
  		    int matStp  = srcDOF*trgDOF;
  		    int denStp = srcDOF;
  		    int chkStp   = trgDOF;
  
  		    real_t newalpha = nrmfc;
  		    for(int i=0; i<trgDOF; i++)
	 	        for(int k=0; k<srcDOF; k++) {
			        int matOff = k*trgDOF + i;
			        int denOff = k;
			        int chkOff = i;
              pointwise_mult (effNum/2, matPtr+matOff, matStp, denPtr+denOff, denStp, chkPtr+chkOff, chkStp);
	 	        }
        }
      }
    }
    
    compute_ifft_trg (All_N);

  }
  return 0;
}

int
pointwise_mult (int n, FFT_COMPLEX* x, int ix, FFT_COMPLEX* y, int iy, FFT_COMPLEX* z, int iz)
{
  int i;
  
  for (i = 0; i < n; i++) {
	  (*z)[0] += ( (*x)[0] * (*y)[0] - (*x)[1] * (*y)[1]);
	  (*z)[1] += ( (*x)[0] * (*y)[1] + (*x)[1] * (*y)[0]);
	  x = x + ix;
	  y = y + iy;
	  z = z + iz;
  }
  return 0;
}

/* ------------------------------------------------------------------------
 */

int
wlist_calc__cpu (FMMWrapper_t *F)
{
  int i, j;
  int srcDOF = 1;
  real_t r;
  Point3 c;
  Node* sl_pos;
  AllNodes *All_N = F->AN;
  Pos* SP = All_N->SP;
  vector<NodeTree>& nodeVec = *All_N->N;

  sl_pos = (Node *) _mm_malloc(sizeof(Node), IDEAL_ALIGNMENT);
    
  sl_pos->x = (real_t *) reals_alloc__aligned (SP[UE].n);
  sl_pos->y = (real_t *) reals_alloc__aligned (SP[UE].n);
  sl_pos->z = (real_t *) reals_alloc__aligned (SP[UE].n);
  sl_pos->num_pts = SP[UE].n;

  for (i = 0; i < nodeVec.size(); i++) {
	  if( nodeVec[i].tag & LET_TRGNODE) { 
	    if( nodeVec[i].child == -1) { 
        for (j = 0; j < nodeVec[i].Wnodes.size(); j++) {
	        int src = nodeVec[i].Wnodes[j];
			    if(nodeVec[src].child == -1 && nodeVec[src].srcNum*srcDOF<pln_size(UE, SP)) { 
				    //S2T - source -> target
	          ulist__direct_evaluation (All_N->Nt[i], All_N->Ns[src]);
			    } 
          else {
				    //M2T - multipole -> target
            c = center (src, nodeVec);
            r = radius (src, nodeVec);
            compute_localpos (c, r, sl_pos, &SP[UE]);
            sl_pos->den_pot = &All_N->src_upw_equ_den[src*pln_size(UE, SP)];
            ulist__direct_evaluation (All_N->Nt[i], *sl_pos);
			    }
		    }
		  }
	  }
  }
  return 0;
}

/* ------------------------------------------------------------------------
 */

int
xlist_calc__cpu (FMMWrapper_t* F)
{
  int i, j;
  int trgDOF = 1;
  Point3 c;
  real_t r;
  Node* tl_pos;
  AllNodes *All_N = F->AN;
  Pos* SP = All_N->SP;
  vector<NodeTree>& nodeVec = *All_N->N;

  tl_pos = (Node *) _mm_malloc(sizeof(Node), IDEAL_ALIGNMENT);
    
  tl_pos->x = (real_t *) reals_alloc__aligned (SP[DC].n);
  tl_pos->y = (real_t *) reals_alloc__aligned (SP[DC].n);
  tl_pos->z = (real_t *) reals_alloc__aligned (SP[DC].n);
  tl_pos->num_pts = SP[DC].n;

  for (i = 0; i < nodeVec.size(); i++) {
	  if( nodeVec[i].tag & LET_TRGNODE) { 
      for (j = 0; j < nodeVec[i].Xnodes.size(); j++) {
	      int src = nodeVec[i].Xnodes[j];
			  if(nodeVec[i].child == -1 && nodeVec[i].trgNum*trgDOF<pln_size(DC, SP)) {
	        ulist__direct_evaluation (All_N->Nt[i], All_N->Ns[src]);
		    } 
        else {
			    //S2L - source -> local
          c = center (i, nodeVec);
          r = radius (i, nodeVec);
          compute_localpos (c, r, tl_pos, &SP[DC]);
          tl_pos->den_pot = &All_N->trg_dwn_chk_val[i*pln_size(DC, SP)];
 
          ulist__direct_evaluation (*tl_pos, All_N->Ns[src]);
		    }
		  }
	  }
  }
  return 0;
}

/* ------------------------------------------------------------------------
 */
int
down_calc__cpu (FMMWrapper_t* F)
{
  real_t r;
  Point3 c;
  real_t *tval;
  real_t *dden;
  Node* sl_pos;
  AllNodes *All_N = F->AN;
  Pos* SP = All_N->SP;
  Trans_matrix* TM = All_N->TM;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* Pre-compute DC2DE matrix */
  compute_DC2DE_mat (TM, SP);

  /* Pre-compute DE2DC matrices */
  TM->DE2DC = (real_t **) malloc (sizeof(real_t*) * 2*2*2);

	for(int a=0; a<2; a++) {
	  for(int b=0; b<2; b++) {
			for(int c=0; c<2; c++) {
        Index3 idx(a, b, c);
        compute_DE2DC_mat (idx, TM, SP);
      }
    }
  }


  for (int i = 0; i < nodeVec.size(); i++) {
    if (nodeVec[i].tag & LET_TRGNODE) {
      if (nodeVec[i].depth >= 3) {
		    int parent = nodeVec[i].parent;
		    Index3 cidx(nodeVec[i].path2Node-2 * nodeVec[parent].path2Node);
		    //L2L - local -> local
        tval = &All_N->trg_dwn_chk_val[i*pln_size(DC, SP)];
        dden = &All_N->trg_dwn_equ_den[parent*pln_size(DE, SP)];
	      int srcDOF = 1;
	      real_t *tmpDen = (real_t *) reals_alloc__aligned (srcDOF*SP[DE].n);	 
        reals_zero (SP[DE].n, tmpDen);
        int l = nodeVec[parent].depth;
        real_t degVec[1]; degVec[0] = 1;
	      real_t sclvec[srcDOF];	 
        for(int s=0; s<srcDOF; s++)		
          sclvec[s] = pow(2.0, l*degVec[s]);
	      int cnt = 0;
	      for(int j=0; j < SP[DE].n; j++)
		      for(int s=0; s < srcDOF; s++) {
		        tmpDen[cnt] = dden[cnt] * sclvec[s];
            cnt++;
          }
        real_t *DE2DCii = TM->DE2DC[cidx(2)+cidx(1)*2+cidx(0)*2*2];
	      dgemv(TM->n, TM->m, 1.0, DE2DCii, tmpDen, 1.0, tval);
		  }

      if (nodeVec[i].depth >= 2) {
		    //L2L - local -> local
        tval = &All_N->trg_dwn_chk_val[i*pln_size(DC, SP)];
        dden = &All_N->trg_dwn_equ_den[i*pln_size(DE, SP)];
	      int srcDOF = 1;
	      real_t *tmpDen = (real_t *) reals_alloc__aligned (srcDOF*SP[DE].n);	 
        reals_zero (SP[DE].n, tmpDen);
	      dgemv(TM->m, TM->n, 1.0, TM->DC2DE, tval, 1.0, tmpDen);
	 
        //scale
        int l = nodeVec[i].depth;
        real_t degVec[1]; degVec[0] = 1;
	      real_t sclvec[srcDOF];	 
        for(int s=0; s<srcDOF; s++)		
          sclvec[s] = pow(2.0, -l*degVec[s]);
	      int cnt = 0;
	      for(int j=0; j < SP[DE].n; j++)
		      for(int s=0; s < srcDOF; s++) {
		        dden[cnt] = dden[cnt] + tmpDen[cnt] * sclvec[s];
		        cnt++;
	        }	  
      }
	  }
  }
  
  sl_pos = (Node *) _mm_malloc(sizeof(Node), IDEAL_ALIGNMENT);
    
  sl_pos->x = (real_t *) reals_alloc__aligned (SP[DE].n);
  sl_pos->y = (real_t *) reals_alloc__aligned (SP[DE].n);
  sl_pos->z = (real_t *) reals_alloc__aligned (SP[DE].n);
  sl_pos->num_pts = SP[DE].n;

  // leaf
  for (int i = 0; i < nodeVec.size(); i++) {
    if (nodeVec[i].tag & LET_TRGNODE) {
      if (nodeVec[i].child == -1) {
		    //L2T - local -> target
        c = center (i, nodeVec);
        r = radius (i, nodeVec);
        compute_localpos (c, r, sl_pos, &SP[DE]);
        sl_pos->den_pot = &All_N->trg_dwn_equ_den[i*pln_size(DE, SP)];
        
        ulist__direct_evaluation (All_N->Nt[i], *sl_pos);
		  }
	  }
  }

  return 0;
}

/* ------------------------------------------------------------------------
 */

int
copy_trg_val__cpu (FMMWrapper_t* F)
{
  int i;
  AllNodes *All_N = F->AN;
  vector<NodeTree>& nodeVec = *All_N->N;
  
  for (i = 0; i < nodeVec.size(); i++) 
	  if( nodeVec[i].tag & LET_TRGNODE)  
      if (nodeVec[i].child == -1) {
        set_value (nodeVec[i].trgNum, All_N->pot_orig, All_N->Nt[i].den_pot, nodeVec[i].trgOwnVecIdxs);
		  }
	   
  return 0;
}

/* ------------------------------------------------------------------------
 */
