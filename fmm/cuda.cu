#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cutil_inline.h>

#include "evaluate.h"
#include "util.h"
#include "reals.h"
#include "../timing/timing.h"
#include "node_gpu.h"

#define NP_3 0
#define NP_4 0
#define NP_6 1

#include "partial.h"

  void
gpu_check_error__srcpos (FILE* fp, const char* filename, size_t line)
{
  cudaError_t C_E = cudaGetLastError ();
  if (C_E) {
    fprintf (fp, "*** [%s:%lu] CUDA ERROR %d: %s ***\n",
        filename, line,
        C_E, cudaGetErrorString (C_E));
    fflush (fp);
    exit (-1); /* abort program */
  }
}

/* ------------------------------------------------------------------------
 */
  int
get_thread_block_size_ulist()
{
  return getenv__int("TBSIZE_ULIST", 128);
}

/* ------------------------------------------------------------------------
 */
  int
get_thread_block_size_up()
{
  return getenv__int("TBSIZE_UP", 128);
}

/* ------------------------------------------------------------------------
 */

  int
get_thread_block_size_up_reduce()
{
	if(NP_3) {
	  return getenv__int("TBSIZE_UP_REDUCE", 256);
	} else if (NP_4) {
	  return getenv__int("TBSIZE_UP_REDUCE", 256);
	} else if (NP_6) {
	  return getenv__int("TBSIZE_UP_REDUCE", 256);
	}
}

/* ------------------------------------------------------------------------
 */

  int
get_thread_block_size_fft_trans()
{
  return getenv__int("TBSIZE_FFT_TRANS", 256);
}

/* ------------------------------------------------------------------------
 */

  int
get_thread_block_size_vlist()
{
  return getenv__int("TBSIZE_VLIST", 128);
}

/* ------------------------------------------------------------------------
 */

  int
get_thread_block_size_down()
{
  return getenv__int("TBSIZE_DOWN", 256);
}

/* ------------------------------------------------------------------------
 */

  int
get_thread_block_size_down_leaf()
{
  return getenv__int("TBSIZE_DOWN_LEAF", 128);
}
/* ------------------------------------------------------------------------
 */

int
get_thread_block_size_wlist()
{
  return getenv__int("TBSIZE_WLIST", 256);
}


__global__
  void
up_eval__gpu (int n_boxes_, int *Bptr_, int *Bn_, 
    dtype *x_, dtype *y_, dtype *z_, dtype *w_,
    dtype *radius_, dtype *c0_, dtype *c1_, dtype *c2_,
    int sp_uc_size, int sp_uc_size_padded, dtype* sp_uc_,
    int uc2ue_r, int uc2ue_r_padded, int uc2ue_c, dtype *uc2ue_,
    /* int num_non_leaf_nodes, */ int *depth_,
    dtype* src_upw_equ_den_)
{
  int i, j;
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  if(bid < n_boxes_) {
    int start, end;

    /* beginning and ending points for this box/node */
    start = Bptr_[bid];
    end = Bptr_[bid] + Bn_[bid];

    /* do it only if this box/node is a leaf node */
    if(start < end) {
      __shared__ dtype potential[BLK_SIZE_UP];
      /* each thread is responsible for at least 1 point in tl_pos */
      /* there are SP[UC].n points in tl_pos */
      /* for each point in tl_pos, iterate over all source points
         corresponding to this source leaf node */
      for(i = tid; i < sp_uc_size; i += blockDim.x) {
        /* there are variables required for each point in tl_pos */
        dtype c0 = c0_[bid];
        dtype c1 = c1_[bid];
        dtype c2 = c2_[bid];
        dtype r = radius_[bid];

        dtype sp_x = sp_uc_[i];
        dtype sp_y = sp_uc_[1 * sp_uc_size_padded + i];
        dtype sp_z = sp_uc_[2 * sp_uc_size_padded + i];

        sp_x = c0 + r * sp_x;
        sp_y = c1 + r * sp_y;
        sp_z = c2 + r * sp_z;
        dtype sp_d = 0.0;

        /* now loop over all points in source leaf node */
        for(j = start; j < end; j++) {
          /* load source's x, y, z, and w */
          dtype x = x_[j]; 
          dtype y = y_[j]; 
          dtype z = z_[j]; 
          dtype w = w_[j];

          x = sp_x - x;
          y = sp_y - y;
          z = sp_z - z;
          dtype rsq = (x * x) + (y * y) + (z * z);
          rsq = rsqrt (rsq);
          sp_d += w * rsq;
        }
        potential[i] = OOFP_R * sp_d;
      }
      __syncthreads();

		
      /* do dgemv */
      dtype td = 0.0;
      dtype sclvec = depth_[bid];
      sclvec = __powf (2.0, -1 * sclvec); /* it's okay to do this in float */
      for(i = tid; i < uc2ue_r; i += blockDim.x) {
        td = 0.0;
        for(j = 0; j < uc2ue_c; j++) {
          td += uc2ue_[j * uc2ue_r_padded + i] * potential[j];
        }


        /* scale */
        td = td * sclvec;
        src_upw_equ_den_[bid * uc2ue_r_padded + i] = td;
      }

    }
  }
}

/* ------------------------------------------------------------------------
 */

__global__
  void
up_eval__gpu_reduction (int num_thr_per_child, 
    int n_boxes_, int reduction_offset, int node_depth,
    int *children,
    dtype *src_upw_equ_den_, int uc2ue_r_padded,
    dtype *ue2uc_, int ue2uc_r, int ue2uc_r_padded, 
    int ue2uc_c,
    dtype *uc2ue_, int uc2ue_r, int uc2ue_c)
{
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  if(bid < n_boxes_) {
    bid += reduction_offset;
    int i, j, k;
    /* this should be 8 * uc2ue_r_padded (SP[UE]) */
    /* NP=3 ==> 256 */
    /* NP=4 ==> 512 */
    __shared__ dtype tmpDen[UC2UE_R_PADDED];
    /* size should be 8x ue2uc_r_padded (SP[UC]) */
    /* NP=3 ==> 1024 */
    /* NP=4 ==> 1280 */
    __shared__ dtype tmpTl[UE2UC_R_PADDED];

    /* id of the child node/box of current node/box */
    /* it should be 0~7 as this is an octree */
    int child_id = tid / num_thr_per_child;
    /* thread ID for the child the thread is responsible for */
    /* it should be anywhere from 0~(num_thr_per_child-1) */
    int child_tid = tid % num_thr_per_child;

    /* first, process current node's child's children */
    /* 1. Child 0 goes through each of ITS 8 children and calculate uden.
     * 2. Calculate Child 0's uden 
     * 3. Calculate the current box/node's uden
     */

    /* Identify the real node IDs for the children nodes */
    int cur_child_id = children[bid] + child_id;

    /* each of these children nodes have 8 children of its own and they SHOULD
     * all be leaf nodes
     */
    /* needed variables */
    int ue2uc_index = 0;
    for(i = tid; i < UE2UC_R_PADDED; i += blockDim.x) {
      tmpTl[i] = 0.0;
    }
    __syncthreads ();

    /* now, go through the 8 children */
    for(i = children[cur_child_id]; i < children[cur_child_id] + 8; i++) {
      dtype* den = &src_upw_equ_den_[i * uc2ue_r_padded];
      dtype sclvec = __powf (2.0, (node_depth + 2)); 

      /* scale */	
      for(j = child_tid; j < uc2ue_r_padded; j += num_thr_per_child) {
        tmpDen[child_id * uc2ue_r_padded + j] = den[j] * sclvec;
      }
      __syncthreads();

      dtype *UE2UCii = &ue2uc_[ue2uc_index * ue2uc_r_padded * ue2uc_c];
      /* dgemv */
      for(j = child_tid; j < ue2uc_r_padded; j += num_thr_per_child) {
        dtype tmp = tmpTl[child_id * ue2uc_r_padded + j];
        for(k = 0; k < ue2uc_c; k++) {
          tmp += UE2UCii[k * ue2uc_r_padded + j] * 
            tmpDen[child_id * uc2ue_r_padded + k];
        }
        tmpTl[child_id * ue2uc_r_padded + j] = tmp;
      }
      ue2uc_index++;
    }
    __syncthreads ();	


    /* At this point, each block of num_thr_per_child 
     * has accumulated the results of its 8 leaf nodes
     * Thus, there are 8 sets of tl_pos, each of which belongs to a block
     * Now, we must do dgemv with UC2UE matrix and tl_pos
     */
    dtype sclvec = __powf (2.0, -(node_depth + 1));
    dtype sclvec_ = __powf (2.0, (node_depth + 1));
    dtype tmp;
    for(i = child_tid; i < uc2ue_r_padded; i += num_thr_per_child) {
      tmp = 0.0;
      for(j = 0; j < uc2ue_c; j++) {
        tmp += uc2ue_[j * uc2ue_r_padded + i] * 
          tmpTl[child_id * ue2uc_r_padded + j];
      }
      tmp = tmp * sclvec;

      src_upw_equ_den_[cur_child_id * uc2ue_r_padded + i] = tmp;
      tmpDen[child_id * uc2ue_r_padded + i] = tmp * sclvec_;
    }

    /* Now that we have the uden for all 8 of its children, do the actual
     * computation required for current box/node 
     */
    /* Do each block of num_thr_per_child do dgemv for its own uden and 
     * UE2UC matrix */
    dtype* UE2UCii = &ue2uc_[child_id * ue2uc_r_padded * ue2uc_c];
    for(i = child_tid; i < ue2uc_r_padded; i += num_thr_per_child) {
      dtype tmp_r = 0.0;
      for(j = 0; j < ue2uc_c; j++) {
        tmp_r += UE2UCii[j * ue2uc_r_padded + i] * 
          tmpDen[child_id * uc2ue_r_padded + j];
      }
      tmpTl[child_id * ue2uc_r_padded + i] = tmp_r;
    }
    __syncthreads ();

    /* reduce the 8 results of degmv on tmpTl (8 * ue2uc_r_padded) */
    /* 4 and 4 */
    for(i = tid; i < UE2UC_R_PADDED / 2; i += blockDim.x) {
      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 2)];
    }
    __syncthreads ();
    /* 2 and 2 */
    for(i = tid; i < UE2UC_R_PADDED / 4; i += blockDim.x) {
      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 4)];
    }
    __syncthreads ();
    /* 1 and 1 */
    for(i = tid; i < UE2UC_R_PADDED / 8; i += blockDim.x) {
      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 8)];
    }
    __syncthreads ();


    /* Finally do dgemv with UC2UE matrix */
    sclvec = __powf (2.0, -node_depth);
    for(i = tid; i < uc2ue_r_padded; i += blockDim.x) {
      tmp = 0.0;
      for(j = 0; j < uc2ue_c; j++) {
        tmp += uc2ue_[j * uc2ue_r_padded + i] * tmpTl[j];
      }

      tmp = tmp * sclvec;

      src_upw_equ_den_[bid * uc2ue_r_padded + i] = tmp;
    }	
  }
}

/* ------------------------------------------------------------------------
 */

__global__
  void
up_eval__gpu_reduction_last (int num_thr_per_child, int n_boxes_, 
    int reduction_offset, int node_depth,
    int *children, 
    dtype *src_upw_equ_den_, int uc2ue_r_padded,
    dtype *ue2uc_, int ue2uc_r, int ue2uc_r_padded, 
    int ue2uc_c,
    dtype *uc2ue_, int uc2ue_r, int uc2ue_c)

{
  int i, j;

  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  if(bid < n_boxes_) {
    bid += reduction_offset;

    __shared__ dtype tmpDen[UC2UE_R_PADDED];
    __shared__ dtype tmpTl[UE2UC_R_PADDED];

    /* 0 ~ 7 */
    int child_id = tid / num_thr_per_child;
    /* 0 ~ num_thr_per_child */
    int child_tid = tid % num_thr_per_child;

    /* node ID of one of 8 children */
    int cur_child_id = children[bid] + child_id;

    /* src_upw_equ_den for the child */
    dtype *den = &src_upw_equ_den_[cur_child_id * uc2ue_r_padded];

    /* scale */
    dtype sclvec = __powf (2.0, (node_depth + 1));
    dtype tmp; 
    for(i = child_tid; i < uc2ue_r_padded; i+= num_thr_per_child) {
      tmp = den[i];
      tmpDen[child_id * uc2ue_r_padded + i] = tmp * sclvec;
    }
    __syncthreads ();

    /* dgemv */
    dtype *UE2UCii = &ue2uc_[child_id * ue2uc_r_padded * ue2uc_c];
    for(i = child_tid; i < ue2uc_r_padded; i += num_thr_per_child) {
      dtype tmp_r = 0.0;
      for(j = 0; j < ue2uc_c; j++) {
        tmp_r += UE2UCii[j * ue2uc_r_padded + i] *
          tmpDen[child_id * uc2ue_r_padded + j];
      }
      tmpTl[child_id * ue2uc_r_padded + i] = tmp_r;
    }
    __syncthreads ();

    /* reduce the 8 children */
    /* 4 and 4 */
    for(i = tid; i < UE2UC_R_PADDED / 2; i += blockDim.x) {
      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 2)];
    }
    __syncthreads ();
    /* 2 and 2 */
    for(i = tid; i < UE2UC_R_PADDED / 4; i += blockDim.x) {
      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 4)];
    }
    __syncthreads ();
    /* 1 and 1 */
    for(i = tid; i < UE2UC_R_PADDED / 8; i += blockDim.x) {
      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 8)];
    }
    __syncthreads ();

    /* final dgemv and scale */
    sclvec = __powf (2.0, -node_depth);
    for(i = tid; i < uc2ue_r_padded; i += blockDim.x) {
      tmp = 0.0;
      for(j = 0; j < uc2ue_c; j++) {
        tmp += uc2ue_[j * uc2ue_r_padded + i] * tmpTl[j];
      }

      tmp = tmp * sclvec;

      src_upw_equ_den_[bid * uc2ue_r_padded + i] = tmp;
    }	
  }
}

__global__
void
up_eval__gpu_non_leaf (int num_thr_per_child, 	
											 int n_boxes_,
											 int node_depth,
											 int *children,
											 dtype *src_upw_equ_den_,
											 int uc2ue_r, int uc2ue_c,
											 int uc2ue_r_padded,
											 dtype *uc2ue_,
											 int ue2uc_r, int ue2uc_c,
											 int ue2uc_r_padded,
											 dtype *ue2uc_,
											 int *tag_,
											 int *depth_)
{
  int i, j;

  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

	__shared__ dtype tmpDen[UC2UE_R_PADDED];
	__shared__ dtype tmpTl[UE2UC_R_PADDED];

	if(bid < n_boxes_) {
		if(depth_[bid] == node_depth && children[bid] != -1) {
	    /* 0 ~ 7 */
	    int child_id = tid / num_thr_per_child;

	    /* 0 ~ num_thr_per_child */
	    int child_tid = tid % num_thr_per_child;

	    /* node ID of one of 8 children */
	    int cur_child_id = children[bid] + child_id;

			dtype *den = &src_upw_equ_den_[cur_child_id * uc2ue_r_padded];

		  dtype sclvec = __powf (2.0, (node_depth + 1));
			dtype tmp;


			if(tag_[cur_child_id] & LET_SRCNODE) {
				/* scale */
		    for(i = child_tid; i < uc2ue_r_padded; i+= num_thr_per_child) {
		      tmp = den[i];
		      tmpDen[child_id * uc2ue_r_padded + i] = tmp * sclvec;
		    }

				/* dgemv */
		    dtype *UE2UCii = &ue2uc_[child_id * ue2uc_r_padded * ue2uc_c];
		    for(i = child_tid; i < ue2uc_r_padded; i += num_thr_per_child) {
		      dtype tmp_r = 0.0;
		      for(j = 0; j < ue2uc_c; j++) {
		        tmp_r += UE2UCii[j * ue2uc_r_padded + i] *
		          			 tmpDen[child_id * uc2ue_r_padded + j];
		      }
		      tmpTl[child_id * ue2uc_r_padded + i] = tmp_r;
		    }
			} else {
				for(i = child_tid; i < ue2uc_r_padded; i += num_thr_per_child) {
					tmpTl[child_id * ue2uc_r_padded + i] = 0.0;
				}
			} /* child == SRC */
			__syncthreads ();


	    /* reduce the 8 children */
	    /* 4 and 4 */
	    for(i = tid; i < UE2UC_R_PADDED / 2; i += blockDim.x) {
	      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 2)];
	    }
	    __syncthreads ();
	    /* 2 and 2 */
	    for(i = tid; i < UE2UC_R_PADDED / 4; i += blockDim.x) {
	      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 4)];
	    }
	    __syncthreads ();
	    /* 1 and 1 */
	    for(i = tid; i < UE2UC_R_PADDED / 8; i += blockDim.x) {
	      tmpTl[i] += tmpTl[i + (UE2UC_R_PADDED / 8)];
	    }
	    __syncthreads ();


	    /* final dgemv and scale */
	    sclvec = __powf (2.0, -node_depth);
	    for(i = tid; i < uc2ue_r; i += blockDim.x) {
	      tmp = 0.0;
	      for(j = 0; j < uc2ue_c; j++) {
	        tmp += uc2ue_[j * uc2ue_r_padded + i] * tmpTl[j];
	      }

	      tmp = tmp * sclvec;

	      src_upw_equ_den_[bid * uc2ue_r_padded + i] = tmp;
	    }
		} /* depth == node_depth */
	} /* bid < n_boxes_ */
}

/* ------------------------------------------------------------------------
 */
void
up_calc__gpu_reduction(FMMWrapper_t *f)
{
	int i;

  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	const int NB = get_thread_block_size_up_reduce ();
	const int NG = nodeVec.size ();
	const int num_thr_per_child = NB / 8;

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);


	/* go through each depth and do up_calc */
	for(i = f->tree_max_depth - 1; i >= 0; i--) {
		up_eval__gpu_non_leaf <<<GB, TB>>> (num_thr_per_child,
																				nodeVec.size (),
																				i, 
																				f->child_d_,
																				f->SRC_UPW_EQU_DEN_d_,
																				f->UC2UE_r, f->UC2UE_c,
																				f->UC2UE_r_padded,
																				f->UC2UE_d_,
																				f->UE2UC_r, f->UE2UC_c,
																				f->UE2UC_r_padded,
																				f->UE2UC_d_,
																				f->tag_d_,
																				f->depth_d_);
	}
	cudaThreadSynchronize ();
	gpu_check_error (stderr);
}

/* ------------------------------------------------------------------------
 */
  int
up_calc__gpu (FMMWrapper_t* f)
{

  Boxes__gpu__t* S;
  assert (f);
  S = &f->S_d_;

  const int NB = get_thread_block_size_up ();
  const int NG = S->n_boxes_;

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

  up_eval__gpu <<<GB, TB>>> (S->n_boxes_, S->Bptr_, S->Bn_, 
      S->x_, S->y_, S->z_, S->w_,
      f->radius_d_, 
      f->center0_d_, f->center1_d_, f->center2_d_,
      f->SP_UC_size, f->SP_UC_size_padded, f->SP_UC_d_,
      f->UC2UE_r, f->UC2UE_r_padded, f->UC2UE_c, 
      f->UC2UE_d_,
      /* f->num_non_leaf_nodes, */ f->depth_d_,
      f->SRC_UPW_EQU_DEN_d_);

  cudaThreadSynchronize ();
  gpu_check_error (stderr);

  up_calc__gpu_reduction (f);
  return 0;
}

/* ------------------------------------------------------------------------
 */
/* ULIST FUNCTIONS */
__global__
  void
ulist_eval__gpu (int n_boxes__trg, int *Bptr__trg, int *Bn__trg, 
    dtype *x__trg, dtype *y__trg, dtype *z__trg, dtype *w__trg,
    int n_boxes__src, int *Bptr__src, int *Bn__src,
    dtype *x__src, dtype *y__src, dtype *z__src, dtype *w__src,
    int* Ptr__u, int* L__u)
{
  /* thread and block ID's */
  const int tid = threadIdx.x;
  const int bid = blockIdx.y * gridDim.x + blockIdx.x;


  if(bid < n_boxes__trg) {

    int i, j, k;

    /* points to beginning and end of this thread block's (target's) ulist 
       neighbor */
    const int u_begin = Ptr__u[bid];
    const int u_end = Ptr__u[bid + 1];

    /* points to first and last point in this thread block (target) */
    const int trg_begin = Bptr__trg[bid];
    const int trg_end = Bptr__trg[bid] + Bn__trg[bid];

    /* Loop over each target point */
    for(i = trg_begin + tid; i < trg_end; i += blockDim.x) {
      dtype xt = x__trg[i];
      dtype yt = y__trg[i];
      dtype zt = z__trg[i];
      dtype wt = 0.0;

      /* For each target point, loop over the soure boxes in the ulist */
      for(j = u_begin; j < u_end; j++) {
        const int src_id = L__u[j];
        const int src_begin = Bptr__src[src_id];
        const int src_end = Bptr__src[src_id] + Bn__src[src_id];

        /* Loop over points in each source box */
        for(k = src_begin; k < src_end; k++) {
          dtype xs = x__src[k];
          dtype ys = y__src[k];
          dtype zs = z__src[k];
          dtype ws = w__src[k];

          xs = xt - xs;
          ys = yt - ys;
          zs = zt - zs;

          dtype rsq = xs * xs + ys * ys + zs * zs;
          rsq = rsqrt (rsq);

          wt += ws * rsq;
        }
      }
      w__trg[i] = OOFP_R * wt;
    }
  }
}

/* ------------------------------------------------------------------------
 */
  int
ulist_calc__gpu (FMMWrapper_t* f)
{
  /* Source and target boxes on GPU */
  const Boxes__gpu__t* S;
  Boxes__gpu__t* T;

  /* Ulist on GPU */
  const UList__gpu__t* U;

  assert (f);

  S = &f->S_d_;
  T = &f->T_d_;
  U = &f->U_d_;

  const int NB = get_thread_block_size_ulist ();
  const int NG = T->n_boxes_;
  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

  ulist_eval__gpu <<<GB, TB>>> (T->n_boxes_, T->Bptr_, T->Bn_, 
      T->x_, T->y_, T->z_, T->w_,
      S->n_boxes_, S->Bptr_, S->Bn_, 
      S->x_, S->y_, S->z_, S->w_,
      U->Ptr_, U->L_);

  // cudaThreadSynchronize ();
  gpu_check_error (stderr);


  return 0;
}
/* ------------------------------------------------------------------------
 */
/* ------------------------------------------------------------------------
 */
/* VLIST FUNCTIONS */
#if 0
/* size should be RP->n =(2np)^3 */
/* NP=3 ==> 224 */
/* NP=4 ==> 896 */
//#define RP_N 216
#define RP_N 512
//#define RP_N 896 

/* incorrectly named */
/* UC2UE_R_PADDED (fake) <= 8 * UC2UE_R_PADDED (real) */
#define UC2UE_R (UC2UE_R_PADDED/8)
#endif

__global__
  void
compute_fft_src__gpu_eval(int n_boxes_, int *depth, int np,
    dtype *src_upw_equ_den_, int uc2ue_r_padded,
    dtype *reg_den_, int reg_den_size)

{
  __shared__ dtype reg_den[RP_N];
  __shared__ dtype tmp_src[UC2UE_R];
  __shared__ int tmp_index[NP_CUBED_POWER_OF_2];
  const int tid = threadIdx.x;
  const int bid = blockIdx.y * gridDim.x + blockIdx.x;

  if(bid < n_boxes_) {
    int i, j, k;
    /* initialize reg_den 
     * this is needed since all of it will be written back, and those
     * that doesn't have src_upw_equ_den written to it should be 0.0
     */
    for(i = tid; i < reg_den_size; i += blockDim.x) {
      reg_den[i] = 0.0;
    }	
    /* sync not needed */


    int l = (int) __powf (2.0, depth[bid]); /* powf is okay */
    int rgnum = 2 * np;
    int cnt;
    int index;

    dtype *src = &src_upw_equ_den_[bid * uc2ue_r_padded];
    /* load src_upw_equ_den */
    for(i = tid; i < uc2ue_r_padded; i += blockDim.x) {
      tmp_src[i] = src[i];
    }
    __syncthreads ();

    /* compute index for each thread */
    cnt = 0;
    index = 0;
    if(tid == 0) {
      for(i = 0; i < np ; i++) {
        for(j = 0; j < np; j++) {
          for(k = 0; k < np; k++) {
            if(i == 0 || i == np - 1 || j == 0 || j == np - 1 
                || k == 0 || k == np - 1) {
              tmp_index[index] = cnt;
              cnt++;
              /* index is the thread id and cnt is the index into the source
                 array that the thread will be accessing */
            }
            index++;
          }
        }
      }
    }
    __syncthreads ();

    dtype tmp;

    i = tid / (np * np);
    j = (tid % (np * np)) / np;
    k = tid % np;

    /* np^3 - (np-2)^3 threads will be valid */
    if(tid < (np * np * np)) {
      if(i == 0 || i == np - 1 || j == 0 || j == np - 1 
          || k == 0 || k == np - 1) {
        tmp = tmp_src[tmp_index[tid]] * l;

        int rgoff = (k + np / 2) * rgnum * rgnum + ( j + np / 2) * rgnum + 
          (i + np / 2);
        reg_den[rgoff] = tmp;
      }
    }
    __syncthreads ();

    for(i = tid; i < reg_den_size ; i+= blockDim.x) {
      reg_den_[bid * reg_den_size + i] = reg_den[i];
    }
  }
}

  void
compute_fft_src__gpu (FMMWrapper_t *f, AllNodes *All_N)
{
  vector<NodeTree>& nodeVec = *All_N->N;
  const int np = getenv__accuracy ();

  /* source: All_N->src_upw_equ_den[i * pln_size (UE, SP)]; padded, SP[UE] */
  /* result: All_N->eff_den[i * eff_size]; padded, (2+2np)*(2np)*(2np) */

  /* scale src_upw_equ_den into tmp_den */
  /* exec samDen2RegDen (tmp_den => reg_den) 
   * this just expands tmp_den into a larger reg_den array 
   */
  /* store the expanded array in shared memory to do coalesced writes */
  int NB; 
  if(f->UC2UE_r_padded <= 32)
    NB = 32;
  else if(f->UC2UE_r_padded <= 64)
    NB = 64;
  else if(f->UC2UE_r_padded <= 128)
    NB = 128;
  else if(f->UC2UE_r_padded <= 256)
    NB = 256;
  else if(f->UC2UE_r_padded <= 512)
    NB = 512;
  else {
    NB = 1024;
    printf("compute_fft_src__gpu: THIS MIGHT CAUSE A PROBLEM\n");
  }

  const int NG = nodeVec.size ();
  dim3 GB(65535, (NG / 65535) + 1, 1);
  dim3 TB(NB, 1, 1);

  compute_fft_src__gpu_eval <<<GB, TB>>> (nodeVec.size (), 
      f->depth_d_,
      getenv__accuracy (),
      f->SRC_UPW_EQU_DEN_d_, 	
      f->UC2UE_r_padded,
      f->reg_den_d_, f->reg_den_size);

  cudaThreadSynchronize ();
  gpu_check_error (stderr);

	#if MIN_DATA
		cudaFree (f->SRC_UPW_EQU_DEN_d_);
	#endif

	cufftResult ccc;
  cufftHandle plan;
  int nnn[3]; nnn[0] = np * 2; nnn[1] = np * 2; nnn[2] = np * 2;
  cufftPlanMany (&plan, 3, nnn, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, 
      nodeVec.size ());
  ccc = cufftExecD2Z (plan, f->reg_den_d_, (cufftDoubleComplex *) f->vlist_src_d_);

  if(ccc != 0) {
    printf("AAA: %d\n", ccc);
    printf("SUCCESS: %d\n", CUFFT_SUCCESS);
    printf("INVALID PLAN: %d\n", CUFFT_INVALID_PLAN);
    printf("ALLOC FAILED: %d\n", CUFFT_ALLOC_FAILED);
    printf("INVALID TYPE: %d\n", CUFFT_INVALID_TYPE);
    printf("INVALID VALUE: %d\n", CUFFT_INVALID_VALUE);
    printf("INTERNAL ERROR: %d\n", CUFFT_INTERNAL_ERROR);
    printf("EXEC FAILED: %d\n", CUFFT_EXEC_FAILED);
    printf("SETUP FAILED: %d\n", CUFFT_SETUP_FAILED);
    printf("INVALID SIZE: %d\n", CUFFT_INVALID_SIZE);
  }

	cufftDestroy (plan);

}


  __global__
void compute_fft_trans__gpu_eval (int rp_n_, int np, dtype* tt)
{
  int tid = threadIdx.x;
  // int bid = blockIdx.y * gridDim.x + blockIdx.x;
  int i1 = (blockIdx.x % 7) - 3;
  int i2 = (blockIdx.x / 7) - 3;
  int i3 = blockIdx.y - 3;

  int idx;
  dtype R = 1.0;
  dtype step = 2.0 / (np - 1);

  dtype denPosX, denPosY, denPosZ;

  if(abs (i1) > 1 || abs (i2) > 1 || abs (i3) > 1) {
    idx = (i1 + 3) + (i2 + 3) * 7 + (i3 + 3) * 7 * 7;
    denPosX = (dtype) i1 * 2.0 * R;
    denPosY = (dtype) i2 * 2.0 * R;
    denPosZ = (dtype) i3 * 2.0 * R;

    /* daxpy can be skipped because a = 1.0 and y is 0.0 */
    /* chkPosX/Y/Z is same as RP.x/y/z */
    for(int i = tid; i < rp_n_; i += blockDim.x) {
      /* compute RP.x, RP.y, RP.z */
      int ii = i % (2 * np);
      int jj = (i / (2 * np)) % (2 * np);
      int kk = i / (2 * np * 2 * np);

      int gi = (ii < np) ? ii : (ii - 2 * np);
      int gj = (jj < np) ? jj : (jj - 2 * np);
      int gk = (kk < np) ? kk : (kk - 2 * np);

      dtype x = R * gi * step;
      dtype y = R * gj * step;
      dtype z = R * gk * step;


      /* kernel */
      x = denPosX - x;
      y = denPosY - y;
      z = denPosZ - z;
      dtype r = x * x + y * y + z * z;
      r = rsqrt (r);
      tt[idx * rp_n_ + i] = OOFP_R * r;

      // if(idx == 0) printf("%d %f\n", i, OOFP_R * r);
    }
  }

}

  void
compute_fft_trans__gpu (FMMWrapper_t *f, AllNodes *All_N)
{
  const int NB = get_thread_block_size_fft_trans ();
  const int np = getenv__accuracy ();

  dim3 GB ((7*7), 7, 1);
  dim3 TB (NB, 1, 1);

  compute_fft_trans__gpu_eval <<<GB, TB>>> (f->RP_n_, 
      np, 
      f->tt);	
  cudaThreadSynchronize ();
  gpu_check_error (stderr);

	cufftResult ccc;
  cufftHandle plan;
  int nnn[3]; nnn[0] = 2 * np; nnn[1] = 2 * np; nnn[2] = 2 * np;
  cufftPlanMany (&plan, 3, nnn, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, (7 * 7 * 7));
  ccc = cufftExecD2Z (plan, f->tt, (cufftDoubleComplex *) f->vlist_trans_d_);

  if(ccc != 0) {
    printf("AAA: %d\n", ccc);
    printf("SUCCESS: %d\n", CUFFT_SUCCESS);
    printf("INVALID PLAN: %d\n", CUFFT_INVALID_PLAN);
    printf("ALLOC FAILED: %d\n", CUFFT_ALLOC_FAILED);
    printf("INVALID TYPE: %d\n", CUFFT_INVALID_TYPE);
    printf("INVALID VALUE: %d\n", CUFFT_INVALID_VALUE);
    printf("INTERNAL ERROR: %d\n", CUFFT_INTERNAL_ERROR);
    printf("EXEC FAILED: %d\n", CUFFT_EXEC_FAILED);
    printf("SETUP FAILED: %d\n", CUFFT_SETUP_FAILED);
    printf("INVALID SIZE: %d\n", CUFFT_INVALID_SIZE);
  }
	
	cufftDestroy (plan);


}

__global__
  void
compute_ifft_trg__gpu_eval_scale_nrmfc (int n_boxes_, 
    dtype *trg_,
    int vlist_array_size,
    int rp_n_)
{
  int i;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  if(bid < n_boxes_) {
    dtype nrmfc = 1.0 / (dtype) rp_n_;
    for(i = threadIdx.x; i < vlist_array_size; i += blockDim.x) {
      trg_[bid * vlist_array_size + i] = trg_[bid * vlist_array_size + i] * 
        nrmfc;
    }
  }
}

#if 0
/* this should be qual to sp_dc_n_padded */
/* NP=3 ==> 32 */
//#define SP_DC_N 32
#define SP_DC_N 64
#endif

__global__
  void
compute_ifft_trg__gpu_regVal2SamVal(int n_boxes_, int np,
    dtype *reg_den_, int reg_den_size_,
    dtype *trg_dwn_chk_val_, int sp_dc_n_padded)
{
  int bid = blockIdx.y * gridDim.x + blockIdx.x;
  int tid = threadIdx.x;

  __shared__ int tmp_index[NP_CUBED_POWER_OF_2];
  __shared__ dtype tmp_trg[SP_DC_N];

  if(bid < n_boxes_) {
    int i, j, k;

    /* initialize tmp_trg 
     * this is needed since all of it will be written back, and those
     * that doesn't have reg_den_ written to it should be 0.0
     */
    for(i = tid; i < sp_dc_n_padded; i += blockDim.x) {
      tmp_trg[i] = 0.0;
    }
    __syncthreads ();

    int rgnum = 2 * np;

    int cnt = 0;
    int index = 0;
    if(tid == 0) {
      for(i = 0; i < np ; i++) {
        for(j = 0; j < np; j++) {
          for(k = 0; k < np; k++) {
            if(i == 0 || i == np - 1 || j == 0 || j == np - 1
                || k == 0 || k == np - 1) {
              /* 'cnt' is index into the SP[DC].n sized array 'tmp_trg' for 
               * thread 'index' */
              tmp_index[index] = cnt;
              cnt++;
            }
            index++;
          }
        }
      }
    }
    __syncthreads ();

    i = tid / (np * np);
    j = (tid % (np * np)) / np;
    k = tid % np;

    if(tid < (np * np * np)) {
      if(i == 0 || i == np - 1 || j == 0 || j == np - 1
          || k == 0 || k == np - 1) {
        int rgoff = (k + np / 2) * rgnum * rgnum + (j + np / 2) * rgnum + 
          (i + np / 2);
        tmp_trg[tmp_index[tid]] = reg_den_[bid * reg_den_size_ + rgoff];
      }
    }
    __syncthreads ();

    for(i = tid; i < sp_dc_n_padded; i += blockDim.x) {
      trg_dwn_chk_val_[bid * sp_dc_n_padded + i] = tmp_trg[i];
      //if(bid == 0) printf("%d %f\n", i, tmp_trg[i]);
    }
  }
}

  void
compute_ifft_trg__gpu (FMMWrapper_t *f, AllNodes *All_N)
{
  /* ============================================================ */
  /* source: eff_val[288] */
  /* target: trg_dwn_chk_val[26] */
  /* ============================================================ */
  /* scale source nrmfc */
  vector<NodeTree>& nodeVec = *All_N->N;

  const int NG = nodeVec.size ();
  const int NB = get_thread_block_size_vlist ();

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

  compute_ifft_trg__gpu_eval_scale_nrmfc <<<GB, TB>>> (nodeVec.size (),
      f->vlist_trg_d_,
      f->vlist_array_size,
      f->RP_n_);

  cudaThreadSynchronize ();
  gpu_check_error (stderr);


  /* ifft */
  cufftResult ccc;
  const int np = getenv__accuracy ();
  cufftHandle plan;
  int nnn[3]; nnn[0] = 2 * np; nnn[1] = 2 * np; nnn[2] = 2 * np;


  #if 0
  printf("doing FFT in multiple steps\n");
  int fft_size = 1024;
  int num_fft_iter = (nodeVec.size () + fft_size - 1) / fft_size;

  cufftPlanMany (&plan, 3, nnn, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, fft_size);

  int i;
  for(i = 0; i < num_fft_iter - 1; i++) {
		ccc = cufftExecZ2D (plan, (cufftDoubleComplex *) &f->vlist_trg_d_[i * fft_size * f->vlist_array_size], &f->reg_den_d_[i * fft_size * f->reg_den_size]);


    if(ccc != 0) {
     printf("CCC: %d\n", ccc);
     printf("SUCCESS: %d\n", CUFFT_SUCCESS);
     printf("INVALID PLAN: %d\n", CUFFT_INVALID_PLAN);
     printf("ALLOC FAILED: %d\n", CUFFT_ALLOC_FAILED);
     printf("INVALID TYPE: %d\n", CUFFT_INVALID_TYPE);
     printf("INVALID VALUE: %d\n", CUFFT_INVALID_VALUE);
     printf("INTERNAL ERROR: %d\n", CUFFT_INTERNAL_ERROR);
     printf("EXEC FAILED: %d\n", CUFFT_EXEC_FAILED);
     printf("SETUP FAILED: %d\n", CUFFT_SETUP_FAILED);
     printf("INVALID SIZE: %d\n", CUFFT_INVALID_SIZE);
   }
  }

  if(nodeVec.size () % fft_size != 0) {
    cufftDestroy (plan);
    cufftPlanMany (&plan, 3, nnn, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,
                   (nodeVec.size () % fft_size ));
  }

	ccc = cufftExecZ2D (plan, (cufftDoubleComplex *) &f->vlist_trg_d_[i * fft_size * f->vlist_array_size], &f->reg_den_d_[i * fft_size * f->reg_den_size]);

  if(ccc != 0) {
    printf("CCC: %d\n", ccc);    printf("SUCCESS: %d\n", CUFFT_SUCCESS);
    printf("INVALID PLAN: %d\n", CUFFT_INVALID_PLAN);
    printf("ALLOC FAILED: %d\n", CUFFT_ALLOC_FAILED);
    printf("INVALID TYPE: %d\n", CUFFT_INVALID_TYPE);
    printf("INVALID VALUE: %d\n", CUFFT_INVALID_VALUE);
    printf("INTERNAL ERROR: %d\n", CUFFT_INTERNAL_ERROR);
    printf("EXEC FAILED: %d\n", CUFFT_EXEC_FAILED);
    printf("SETUP FAILED: %d\n", CUFFT_SETUP_FAILED);
    printf("INVALID SIZE: %d\n", CUFFT_INVALID_SIZE);
  }
  #endif



	#if 1
  cufftPlanMany (&plan, 3, nnn, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, 
      nodeVec.size ());
  ccc = cufftExecZ2D (plan, (cufftDoubleComplex *) f->vlist_trg_d_, f->reg_den_d_);

  if(ccc != 0) {
    printf("CCC: %d\n", ccc);
    printf("SUCCESS: %d\n", CUFFT_SUCCESS);
    printf("INVALID PLAN: %d\n", CUFFT_INVALID_PLAN);
    printf("ALLOC FAILED: %d\n", CUFFT_ALLOC_FAILED);
    printf("INVALID TYPE: %d\n", CUFFT_INVALID_TYPE);
    printf("INVALID VALUE: %d\n", CUFFT_INVALID_VALUE);
    printf("INTERNAL ERROR: %d\n", CUFFT_INTERNAL_ERROR);
    printf("EXEC FAILED: %d\n", CUFFT_EXEC_FAILED);
    printf("SETUP FAILED: %d\n", CUFFT_SETUP_FAILED);
    printf("INVALID SIZE: %d\n", CUFFT_INVALID_SIZE);
  }
	#endif

	cufftDestroy(plan);

	#if MIN_DATA
		alloc__TRG_DWN_CHK_VAL__ (f);
	#endif


  /* regVal2SamVal */
  int NB_;
  if(f->SP_DC_n_padded_ <= 32)
    NB_ = 32;
  else if(f->SP_DC_n_padded_ <= 64)
    NB_ = 64;
  else if(f->SP_DC_n_padded_ <= 128)
    NB_ = 128;
  else if(f->SP_DC_n_padded_ <= 256)
    NB_ = 256;
  else if(f->SP_DC_n_padded_ <= 512)
    NB_ = 512;
  else {
    NB_ = 1024;
    printf("compute_ifft_src__gpu: THIS MIGHT CAUSE A PROBLEM\n");
  }

  dim3 GB_ (65535, (NG / 65535) + 1, 1);
  dim3 TB_ (NB_, 1, 1);


  compute_ifft_trg__gpu_regVal2SamVal <<<GB_, TB_>>> (nodeVec.size (), np,
      // f->reg_den_ifft_d_,
      f->reg_den_d_,
      f->reg_den_size,
      f->TRG_DWN_CHK_VAL_d_,
      f->SP_DC_n_padded_);

  cudaThreadSynchronize ();
  gpu_check_error (stderr);

}

#define VLIST_SIZE 288

__global__
  void
vlist_calc__gpu_eval (int n_boxes_, dtype *src_, dtype *trans_, dtype *trg_,
    int *vlist_, int* tlist_, int* list_ptr_, 
    int vlist_array_size)
{
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;


  if(bid < n_boxes_) {
    int i, j;
    int vi, ti;

    dtype re, im;
    dtype tmp_src_re, tmp_src_im;
    dtype tmp_trans_re, tmp_trans_im;

    for(i = tid; i < (vlist_array_size / 2); i+= blockDim.x) {
      re = 0.0; im = 0.0;
      for(j = list_ptr_[bid]; j < list_ptr_[bid + 1]; j++) {
        vi = vlist_[j];
        ti = tlist_[j];

        tmp_src_re = src_[vi * vlist_array_size + i * 2 + 0];
        tmp_src_im = src_[vi * vlist_array_size + i * 2 + 1];

        tmp_trans_re = trans_[ti * vlist_array_size + i * 2 + 0];
        tmp_trans_im = trans_[ti * vlist_array_size + i * 2 + 1];


        re += tmp_src_re * tmp_trans_re - tmp_src_im * tmp_trans_im;
        im += tmp_src_re * tmp_trans_im + tmp_src_im * tmp_trans_re;
      }

      trg_[bid * vlist_array_size + i * 2 + 0] = re;
      trg_[bid * vlist_array_size + i * 2 + 1] = im;

    }
  }
}

  void
vlist_calc__gpu_ (FMMWrapper_t *f, AllNodes *All_N)
{
  vector<NodeTree>& nodeVec = *All_N->N;
  const int NB = get_thread_block_size_vlist ();
  const int NG = nodeVec. size();

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

  vlist_calc__gpu_eval <<<GB, TB>>> (nodeVec.size (), f->vlist_src_d_, 
      f->vlist_trans_d_, f->vlist_trg_d_,
      f->vlist_d_, f->tlist_d_, f->vlist_ptr_d_,
      f->vlist_array_size);

  cudaThreadSynchronize ();
  gpu_check_error (stderr);

}

/* ------------------------------------------------------------------------
 */

int
vlist_calc__gpu (FMMWrapper_t *f)
{
  AllNodes *All_N = f->AN;
  compute_fft_src__gpu (f, All_N);
  compute_fft_trans__gpu (f, All_N);
  vlist_calc__gpu_ (f, All_N);
  compute_ifft_trg__gpu (f, All_N);

  return 0;
}
/* ------------------------------------------------------------------------
 */

/* ------------------------------------------------------------------------
 */
#if 0
/* DOWN_CALC FUNCTIONS */
/* NP=3 ==> 256 */
/* NP=4 ==> 512 */
//#define SP_DE_N_PADDED_8 256
#define SP_DE_N_PADDED_8 512
#endif
__global__
  void
down_eval__gpu(int num_thr_per_child, int n_boxes_, int offset, int* children,
    int3 *path2Node, int max_depth,
    dtype *trg_dwn_chk_val_, int sp_dc_n, int sp_dc_n_padded,
    dtype *trg_dwn_equ_den_, int sp_de_n, int sp_de_n_padded,
    dtype *dc2de_, int dc2de_r, int dc2de_r_padded, int dc2de_c,
    dtype *de2dc_, int de2dc_r, int de2dc_r_padded, int de2dc_c)
{
  /* at minimum size of 8x SP[DE].n padded */
  __shared__ dtype tmpDen[SP_DE_N_PADDED_8];
  /* at minimum size of 8x SP[DE].n padded */
  __shared__ dtype tmpDen_[SP_DE_N_PADDED_8];
  __shared__ dtype tmpDen__[SP_DE_N_PADDED_8];

  int i, j, k;	
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  if(bid < n_boxes_) {
    bid += offset;
    /* ---------------------------------------------------------- */
    /* do nodes at depth = 2 */
    int node_depth = 2;
    /* dgemv */
    for(i = tid; i < dc2de_r_padded; i += blockDim.x) {
      dtype tmp = 0.0;
      for(j = 0; j < dc2de_c; j++) {
        tmp += dc2de_[j * dc2de_r_padded + i] * 
          trg_dwn_chk_val_[bid * sp_dc_n_padded + j];
      }
      tmpDen[i] = tmp;
    }

    /* scale */	
    dtype sclvec = __powf (2.0, -node_depth);
    for(i = tid; i < sp_de_n_padded; i += blockDim.x) {
      tmpDen[i] = tmpDen[i] * sclvec;
#if 0
      trg_dwn_equ_den_[bid * sp_de_n_padded + i] = tmpDen[i];	
#endif
    }
    __syncthreads ();
    /* ---------------------------------------------------------- */

    /* ---------------------------------------------------------- */
    /* do depth 3 */
    /* ID of this node's child at depth 3 */
    int child_id = tid / num_thr_per_child;
    /* sub-tid 0~num_thr_per_child for this child child_id */
    int child_tid = tid % num_thr_per_child;
    /* real child ID */
    int cur_child_id = children[bid] + child_id;

    /* parent's dden/trg_dwn_equ_den is in tmpDen */
    /* scale */
    sclvec = __powf (2.0, node_depth);
    for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
      tmpDen_[child_id * sp_de_n_padded + i] = tmpDen[i] * sclvec;
    }

    int3 cidx_parent = path2Node[bid];
    int3 cidx_child = path2Node[cur_child_id];
    int de2dc_index = (cidx_child.z - 2 * cidx_parent.z) + 
      (cidx_child.y - 2 * cidx_parent.y) * 2 +
      (cidx_child.x - 2 * cidx_parent.x) * 4;
    dtype *de2dc_cur = &de2dc_[de2dc_index * de2dc_r_padded * de2dc_c];
    /* dgemv */
    for(i = child_tid; i < de2dc_r_padded; i += num_thr_per_child) {
      dtype tmp = trg_dwn_chk_val_[cur_child_id * sp_dc_n_padded + i];
      for(j = 0; j < de2dc_c; j++) {
        tmp += de2dc_cur[j * de2dc_r_padded + i] * 
          tmpDen_[child_id * sp_de_n_padded + j];
      }
      tmpDen[child_id * de2dc_r_padded + i] = tmp;
    }

    /* dgemv */
    for(i = child_tid; i < dc2de_r_padded; i += num_thr_per_child) {
      dtype tmp = 0.0;
      for(j = 0; j < dc2de_c; j++) {
        tmp += dc2de_[j * dc2de_r_padded + i] * 
          tmpDen[child_id * de2dc_r_padded + j];
      }
      tmpDen_[child_id * dc2de_r_padded + i] = tmp;
    }
    /* scale */
    node_depth++;
    sclvec = __powf (2.0, -node_depth);
    for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
      tmpDen_[child_id * sp_de_n_padded + i] = 
        tmpDen_[child_id * sp_de_n_padded + i] * sclvec;
    }
    /* ---------------------------------------------------------- */

    if(node_depth == max_depth) {
      /* write back the results to main memory for leaf node computation */
      for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
        trg_dwn_equ_den_[cur_child_id * sp_de_n_padded + i] = 
          tmpDen_[child_id * sp_de_n_padded + i];
      }
    } else {
#if 0
      for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
        trg_dwn_equ_den_[cur_child_id * sp_de_n_padded + i] = 
          tmpDen_[child_id * sp_de_n_padded + i];
      }
#endif
      /* continue down to next level (depth 4) */
      /* each set of num_thr_per_child is responsible for all 8 of its children 
       */
      int first_child = children[cur_child_id];
      /* scale */
      sclvec = __powf (2.0, node_depth);
      for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
        tmpDen_[child_id * sp_de_n_padded + i] =  
          tmpDen_[child_id * sp_de_n_padded + i] * sclvec;
      }
      node_depth++;

      /* go through each child */
      for(i = first_child ; i < first_child + 8; i++) {
        cidx_parent = path2Node[cur_child_id];
        cidx_child = path2Node[i];
        de2dc_index = (cidx_child.z - 2 * cidx_parent.z) +
          (cidx_child.y - 2 * cidx_parent.y) * 2 +
          (cidx_child.x - 2 * cidx_parent.x) * 4 ;
        dtype *de2dc_cur = &de2dc_[de2dc_index * de2dc_r_padded * de2dc_c];

        /* dgemv */
        for(j = child_tid; j < de2dc_r_padded; j += num_thr_per_child) {
          dtype tmp = trg_dwn_chk_val_[i * sp_dc_n_padded + j];
          for(k = 0; k < de2dc_c; k++) {
            tmp += de2dc_cur[k * de2dc_r_padded + j] * 
              tmpDen_[child_id * sp_de_n_padded + k];
          }
          tmpDen[child_id * de2dc_r_padded + j] = tmp;
        }


        /* dgemv */
        for(j = child_tid; j < dc2de_r_padded; j += num_thr_per_child) {
          dtype tmp = 0.0;
          for(k = 0; k < dc2de_c; k++) {
            tmp += dc2de_[k * dc2de_r_padded + j] * 	
              tmpDen[child_id * de2dc_r_padded + k];
          }
          tmpDen__[child_id * dc2de_r_padded + j] = tmp;
        }


        /* scale */
        sclvec = __powf (2.0, -node_depth);
        for(j = child_tid; j < sp_de_n_padded; j += num_thr_per_child) {
          trg_dwn_equ_den_[i * sp_de_n_padded + j] = 
            tmpDen__[child_id * sp_de_n_padded + j] * sclvec;
          // if(i==585) printf("%d %f\n", j, trg_dwn_equ_den_[i * sp_de_n_padded + j]);
        }
      }
    }
  }

}


__global__
  void
down_eval__gpu_(int num_thr_per_child, int n_boxes_, int offset, int* children,
    int3 *path2Node, int max_depth,
    dtype *trg_dwn_chk_val_, int sp_dc_n, int sp_dc_n_padded,
    dtype *trg_dwn_equ_den_, int sp_de_n, int sp_de_n_padded,
    dtype *dc2de_, int dc2de_r, int dc2de_r_padded, int dc2de_c,
    dtype *de2dc_, int de2dc_r, int de2dc_r_padded, int de2dc_c)
{
  /* at minimum size of 8x SP[DE].n padded */
  __shared__ dtype tmpDen[SP_DE_N_PADDED_8];
  /* at minimum size of 8x SP[DE].n padded */
  __shared__ dtype tmpDen_[SP_DE_N_PADDED_8];
  __shared__ dtype tmpDen__[SP_DE_N_PADDED_8];

  int i, j, k;	
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;



  if(bid < n_boxes_) {
#if 0
    for(i = tid; i < SP_DE_N_PADDED_8; i += blockDim.x) {
      tmpDen[i] = 0.0;
      tmpDen_[i] = 0.0;
      tmpDen__[i] = 0.0;
    }
    __syncthreads ();
#endif

    bid += offset;
    int node_depth = 4;

    dtype sclvec = __powf (2.0, node_depth);
    for(i = tid; i < sp_de_n_padded; i += blockDim.x) {
      tmpDen[i] = trg_dwn_equ_den_[bid * sp_de_n_padded + i] * sclvec;
      //if(bid==4680) printf("||| %d %f\n", i, tmpDen[i]);
      // if(bid==617) printf("||| %d %f\n", i, trg_dwn_equ_den_[bid * sp_de_n_padded + i]);
    }
    __syncthreads ();


    /* 0 ~ 7 */
    int child_id = tid / num_thr_per_child;
    int child_tid = tid % num_thr_per_child;
    int cur_child_id = children[bid] + child_id;

    int3 cidx_parent = path2Node[bid];
    int3 cidx_child = path2Node[cur_child_id];
    int de2dc_index = (cidx_child.z - 2 * cidx_parent.z) + 
      (cidx_child.y - 2 * cidx_parent.y) * 2 +
      (cidx_child.x - 2 * cidx_parent.x) * 4;
    dtype *de2dc_cur = &de2dc_[de2dc_index * de2dc_r_padded * de2dc_c];
    /* dgemv */
    for(i = child_tid; i < de2dc_r_padded; i += num_thr_per_child) {
      dtype tmp = trg_dwn_chk_val_[cur_child_id * sp_dc_n_padded + i];
      // dtype tmp1 = 0.0;
      for(j = 0; j < de2dc_c; j++) {
        tmp += de2dc_cur[j * de2dc_r_padded + i] * tmpDen[j];
        // tmp1 += de2dc_cur[j * de2dc_r_padded + i] * tmpDen[j];
        /*
           if(cur_child_id==37447 && i==0) 
           printf("%d %f x %f += %f\n", j, de2dc_cur[j * de2dc_r_padded + i],
           tmpDen[j], tmp1);
         */
      }
      // tmpDen_[child_id * de2dc_r_padded + i] = tmp + tmp1;
      tmpDen_[child_id * de2dc_r_padded + i] = tmp;
      // if(cur_child_id==37447) printf("%d %d %d %f %f\n", bid, i, de2dc_index, tmp1, tmp);
    }
    __syncthreads ();

    /* dgemv */
    for(i = child_tid; i < dc2de_r_padded; i += num_thr_per_child) {
      dtype tmp = 0.0;
      for(j = 0; j < dc2de_c; j++) {
        tmp += dc2de_[j * dc2de_r_padded + i] * 
          tmpDen_[child_id * de2dc_r_padded + j];
      }
      tmpDen__[child_id * dc2de_r_padded + i] = tmp;
      // if(cur_child_id==4681) printf("%d %d %f\n", bid, i, tmp);
    }
    __syncthreads ();

    /* scale */
    node_depth++;
    sclvec = __powf (2.0, -node_depth);
    for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
      tmpDen_[child_id * sp_de_n_padded + i] =
        tmpDen__[child_id * sp_de_n_padded + i] * sclvec;	
      // if(cur_child_id==4681) printf("%d %d %f\n", bid, i, tmpDen_[child_id * sp_de_n_padded + i]);
    }
    __syncthreads ();

    /* level 5 is last */
    if(node_depth == max_depth) {
      for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
        trg_dwn_equ_den_[cur_child_id * sp_de_n_padded + i] = 
          tmpDen_[child_id * sp_de_n_padded + i];
        // if(cur_child_id==4681) printf("%d %d %f\n", bid, child_tid, trg_dwn_equ_den_[cur_child_id * sp_de_n_padded + child_tid]);
      }
    } else {
      /* continue down to next level (depth 6) */
      int first_child = children[cur_child_id];

      /* scale */
      sclvec = __powf (2.0, node_depth);
      for(i = child_tid; i < sp_de_n_padded; i += num_thr_per_child) {
        tmpDen_[child_id * sp_de_n_padded + i] = 
          tmpDen_[child_id * sp_de_n_padded + i] * sclvec;
      }
      node_depth++;

      /* go through each child */
      for(i = first_child ; i < first_child + 8; i++) {
        cidx_parent = path2Node[cur_child_id];
        cidx_child = path2Node[i];
        de2dc_index = (cidx_child.z - 2 * cidx_parent.z) +
          (cidx_child.y - 2 * cidx_parent.y) * 2 +
          (cidx_child.x - 2 * cidx_parent.x) * 4 ;
        dtype *de2dc_cur = &de2dc_[de2dc_index * de2dc_r_padded * de2dc_c];

        /* dgemv */
        for(j = child_tid; j < de2dc_r_padded; j += num_thr_per_child) {
          dtype tmp = trg_dwn_chk_val_[i * sp_dc_n_padded + j];
          for(k = 0; k < de2dc_c; k++) {
            tmp += de2dc_cur[k * de2dc_r_padded + j] * 
              tmpDen_[child_id * sp_de_n_padded + k];
          }
          tmpDen[child_id * de2dc_r_padded + j] = tmp;
        }


        /* dgemv */
        for(j = child_tid; j < dc2de_r_padded; j += num_thr_per_child) {
          dtype tmp = 0.0;
          for(k = 0; k < dc2de_c; k++) {
            tmp += dc2de_[k * dc2de_r_padded + j] * 	
              tmpDen[child_id * de2dc_r_padded + k];
          }
          tmpDen__[child_id * dc2de_r_padded + j] = tmp;
        }


        /* scale */
        sclvec = __powf (2.0, -node_depth);
        for(j = child_tid; j < sp_de_n_padded; j += num_thr_per_child) {
          trg_dwn_equ_den_[i * sp_de_n_padded + j] = 
            tmpDen__[child_id * sp_de_n_padded + j] * sclvec;
        }
      }
    }
  }
}

#if 0
/* should be equal to SP[DE].n padded*/
/* NP=3 ==> 32 */
/* NP=3 ==> 64 */
//#define SL_POS_SIZE 32
#define SL_POS_SIZE 64
#endif

__global__
  void
down_eval__gpu_leaf (int n_boxes_, int *Bptr_, int *Bn_,
    dtype *x_, dtype *y_, dtype *z_, dtype *w_,
    dtype *sp_de_, int sp_de_n, int sp_de_n_padded,
    dtype *trg_dwn_equ_den_,
    dtype *radius_, dtype *center0_, dtype *center1_,
    dtype *center2_, int offset)
{
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  if(bid < n_boxes_) {
    __shared__ dtype SL_POS[4][SL_POS_SIZE];

    int start;
    int end;

    /* first and last point in this box */
    start = Bptr_[bid];
    end = start + Bn_[bid];

    /* if there are any points in the box (leaf node) */
    if(start < end) {
      int i, j;
      dtype c0, c1, c2, r;
      /* find center and rdaius */
      c0 = center0_[bid - offset];
      c1 = center1_[bid - offset];
      c2 = center2_[bid - offset];
      r = radius_[bid - offset];
      /* for each thread, do 1 point in sl_pos */
      for(i = tid; i < sp_de_n_padded; i += blockDim.x) {
        SL_POS[0][i] = c0 + r * sp_de_[i];
        SL_POS[1][i] = c1 + r * sp_de_[sp_de_n_padded + i];
        SL_POS[2][i] = c2 + r * sp_de_[2 * sp_de_n_padded + i];
        SL_POS[3][i] = trg_dwn_equ_den_[bid * sp_de_n_padded + i];
      }
      __syncthreads ();

      /* ulist calc */
      /* for each point in target */
      for(i = tid; i < Bn_[bid]; i += blockDim.x) {
        dtype tx = x_[start + i];
        dtype ty = y_[start + i];
        dtype tz = z_[start + i];
        dtype td = 0.0;
        /* for each point in sl_pos */
        for(j = 0; j < sp_de_n; j++) {
          dtype x = tx - SL_POS[0][j];
          dtype y = ty - SL_POS[1][j];
          dtype z = tz - SL_POS[2][j];
          dtype rsq = (x * x) + (y * y) + (z * z);
          rsq = rsqrt (rsq);
          td += SL_POS[3][j] * rsq;
        }
        w_[start + i] += OOFP_R * td;
        // if(bid == 73) printf("%d %f\n", tid, OOFP_R * td);
      }
    }
  }
}

/* ------------------------------------------------------------------------
 */
__global__
void
down_eval__gpu_depth_2 (int n_boxes_,
												int *tag_,
												int sp_dc_n_padded,
												dtype *trg_dwn_chk_val_,
												int dc2de_c, 
												int dc2de_r,
												int dc2de_r_padded,
												dtype *dc2de_,
												int sp_de_n_padded,
												dtype *trg_dwn_equ_den_
											 )
{
	int i, j; 
	int node_depth = 2; 
	int tid = threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;

  __shared__ dtype tmpDen[SP_DE_N_PADDED_8];

	if(bid < n_boxes_) {
		if(tag_[bid] & LET_TRGNODE) {
			/* dgemv */
			for(i = tid; i < dc2de_r_padded; i += blockDim.x) {
				dtype tmp = 0.0;
				for(j = 0; j < dc2de_c; j++) {
					tmp += dc2de_[j * dc2de_r_padded + i] * 
								 trg_dwn_chk_val_[bid * sp_dc_n_padded + j];
				}
				tmpDen[i] = tmp;
			}
			
			/* scale */
			dtype sclvec = __powf (2.0, -node_depth);
			for(i = tid; i < sp_de_n_padded; i+= blockDim.x) {
				trg_dwn_equ_den_[bid * sp_de_n_padded + i] = tmpDen[i] * sclvec;
			}
		}
	}
}

__global__
void
down_eval__gpu_non_leaves (int n_boxes_,
													 int node_depth,
												 	 int *tag_,
													 int *depth_,
													 int3 *path2Node_,
													 int *parent_,
													 int sp_dc_n_padded,
													 dtype *trg_dwn_chk_val_,
													 int dc2de_c, 
													 int dc2de_r,
													 int dc2de_r_padded,
													 dtype *dc2de_,
													 int de2dc_c,
													 int de2dc_r,
													 int de2dc_r_padded,
													 dtype *de2dc_,
													 int sp_de_n_padded,
													 dtype *trg_dwn_equ_den_
											 )

{
	int i, j; 
	int tid = threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;

	__shared__ dtype tmpDen[SP_DE_N_PADDED_8];
	__shared__ dtype tmpDen_[SP_DE_N_PADDED_8];

	if(bid < n_boxes_) {
		if(tag_[bid] & LET_TRGNODE && depth_[bid] == node_depth) {
			int parent_bid = parent_[bid];

			/* scale */
			dtype sclvec = __powf (2.0, (node_depth - 1));
			for(i = tid; i < sp_de_n_padded; i += blockDim.x) {
				tmpDen[i] = trg_dwn_equ_den_[parent_bid * sp_de_n_padded + i] * sclvec;
			}	
			__syncthreads ();


			/* dgemv */
			int3 cidx_parent = path2Node_[parent_bid];
			int3 cidx_child = path2Node_[bid];
			int de2dc_index = (cidx_child.z - 2 * cidx_parent.z) + 
												(cidx_child.y - 2 * cidx_parent.y) * 2 + 
												(cidx_child.x - 2 * cidx_parent.x) * 4;
			dtype *de2dc_cur = &de2dc_[de2dc_index * de2dc_r_padded * de2dc_c]; 
			for(i = tid; i < de2dc_r_padded; i += blockDim.x) {
				dtype tmp = trg_dwn_chk_val_[bid * sp_dc_n_padded + i];
				for(j = 0; j < de2dc_c; j++) {
					tmp += de2dc_cur[j * de2dc_r_padded + i] * tmpDen[j];
				}
				tmpDen_[i] = tmp;
			}
			__syncthreads ();


			/* dgemv */
			for(i = tid; i < dc2de_r; i += blockDim.x) {
				dtype tmp = 0.0;
				for(j = 0; j < dc2de_c; j++) {
					tmp += dc2de_[j * dc2de_r_padded + i] * tmpDen_[j];
				}
				tmpDen[i] = tmp;
			}
			__syncthreads ();


			/* scale and write back */
			sclvec = __powf (2.0, -node_depth);	
			for(i = tid; i < sp_de_n_padded; i += blockDim.x) {
				trg_dwn_equ_den_[bid * sp_de_n_padded + i] = tmpDen[i] * sclvec;	
			}
		}
	}
}


__global__
void
down_eval__gpu_leaves (int n_boxes_,
											 int *Bptr_,
											 int *Bn_,
											 dtype *x_,
											 dtype *y_,
											 dtype *z_,
											 dtype *w_,
											 int sp_de_n,
											 int sp_de_n_padded,
											 dtype *sp_de_,
											 dtype *trg_dwn_equ_den_,
											 dtype *radius_,
											 dtype *center0_,
											 dtype *center1_,
											 dtype *center2_
											 )
{
	int tid = threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;

	int i, j;
	int start, end;

	__shared__ dtype SL_POS[4][SL_POS_SIZE];

	if(bid < n_boxes_) {
		start = Bptr_[bid];
		end = start + Bn_[bid];

		if(start < end) {
			dtype c0, c1, c2, r;
			/* find center and rdaius */
			c0 = center0_[bid];
			c1 = center1_[bid];
			c2 = center2_[bid];
			r = radius_[bid];
			/* for each thread, do 1 point in sl_pos */
			for(i = tid; i < sp_de_n_padded; i += blockDim.x) {
				SL_POS[0][i] = c0 + r * sp_de_[i];
				SL_POS[1][i] = c1 + r * sp_de_[sp_de_n_padded + i];
				SL_POS[2][i] = c2 + r * sp_de_[2 * sp_de_n_padded + i];
				SL_POS[3][i] = trg_dwn_equ_den_[bid * sp_de_n_padded + i];
			}
			__syncthreads ();

			/* ulist calc */
			/* for each point in target */
			for(i = tid; i < Bn_[bid]; i += blockDim.x) {
				dtype tx = x_[start + i];
				dtype ty = y_[start + i];
				dtype tz = z_[start + i];
				dtype td = 0.0;
				/* for each point in sl_pos */
				for(j = 0; j < sp_de_n; j++) {
					dtype x = tx - SL_POS[0][j];
					dtype y = ty - SL_POS[1][j];
					dtype z = tz - SL_POS[2][j];
					dtype rsq = (x * x) + (y * y) + (z * z);
					rsq = rsqrt (rsq);
					td += SL_POS[3][j] * rsq;
				}
				w_[start + i] += OOFP_R * td;
      }
		}		
	}

}

/* ------------------------------------------------------------------------
 */

int
d2d__gpu (FMMWrapper_t *f)
{
	int i;

	AllNodes *All_N = f->AN;
	vector<NodeTree>& nodeVec = *All_N->N;

  /* do level 0~4 
   * nothing is done for level 0 and 1 */
  const int NB = get_thread_block_size_wlist ();
  const int NG = nodeVec.size ();

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

	/* do non-leaves */
	/* first do level 2 */
	down_eval__gpu_depth_2 <<<GB, TB>>> (nodeVec.size (),
																			 f->tag_d_,
																			 f->SP_DC_n_padded_,
																			 f->TRG_DWN_CHK_VAL_d_,
																			 f->DC2DE_c, 
																			 f->DC2DE_r,
																			 f->DC2DE_r_padded,
																			 f->DC2DE_d_,
																			 f->SP_DE_n_padded,
																			 f->TRG_DWN_EQU_DEN_d_);
	cudaThreadSynchronize ();
	gpu_check_error (stderr);


	/* do level 3 ~ last */
	for(i = 3; i <= f->tree_max_depth; i++) {
		/* 	
			input: tval = trg_dwn_chk_val
			output: dden = trg_dwn_equ_den
			algorithm: 	
				tmpDen = trg_dwn_equ_den[parent] * sclvec
				trg_dwn_chk_val[i] += dgemv (DE2DC, tmpDen)
				tmpDen = dgemv (DC2DE, tval[i])
				trg_dwn_equ_den[i] = tmpDen * sclvec
		 */	
		down_eval__gpu_non_leaves <<<GB, TB>>> (nodeVec.size (),
																						i,
																						f->tag_d_,
																						f->depth_d_,
																						f->path2Node_d_,
																						f->parent_d_,
																						f->SP_DC_n_padded_,
																						f->TRG_DWN_CHK_VAL_d_,
																						f->DC2DE_c,
																						f->DC2DE_r,
																						f->DC2DE_r_padded,
																						f->DC2DE_d_,
																						f->DE2DC_c,
																						f->DE2DC_r,
																						f->DE2DC_r_padded,
																						f->DE2DC_d_,
																						f->SP_DE_n_padded,
																						f->TRG_DWN_EQU_DEN_d_);
	  cudaThreadSynchronize ();
	  gpu_check_error (stderr);

	}

	return 0;
}

int
d2t__gpu (FMMWrapper_t *f)
{
	AllNodes *All_N = f->AN;
	vector<NodeTree>& nodeVec = *All_N->N;

  const int NB = get_thread_block_size_wlist ();
  const int NG = nodeVec.size ();

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

	/* do leaves */
	down_eval__gpu_leaves <<<GB, TB>>> (nodeVec.size (),
																			f->T_d_.Bptr_,
																			f->T_d_.Bn_,
																			f->T_d_.x_,
																			f->T_d_.y_,
																			f->T_d_.z_,
																			f->T_d_.w_,
																			f->SP_DE_n_,
																			f->SP_DE_n_padded,
																			f->SP_DE_d_,
																			f->TRG_DWN_EQU_DEN_d_,
																			f->radius_d_,
																			f->center0_d_,
																			f->center1_d_,
																			f->center2_d_
																			);
	cudaThreadSynchronize ();
	gpu_check_error (stderr);

	return 0;
}

int
down_calc__gpu (FMMWrapper_t *f)
{
	int i;

	AllNodes *All_N = f->AN;
	vector<NodeTree>& nodeVec = *All_N->N;

  /* do level 0~4 
   * nothing is done for level 0 and 1 */
  const int NB = get_thread_block_size_wlist ();
  const int NG = nodeVec.size ();

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

	/* do non-leaves */
	/* first do level 2 */
	down_eval__gpu_depth_2 <<<GB, TB>>> (nodeVec.size (),
																			 f->tag_d_,
																			 f->SP_DC_n_padded_,
																			 f->TRG_DWN_CHK_VAL_d_,
																			 f->DC2DE_c, 
																			 f->DC2DE_r,
																			 f->DC2DE_r_padded,
																			 f->DC2DE_d_,
																			 f->SP_DE_n_padded,
																			 f->TRG_DWN_EQU_DEN_d_);
	cudaThreadSynchronize ();
	gpu_check_error (stderr);

	/* do level 3 ~ last */
	for(i = 3; i <= f->tree_max_depth; i++) {
		/* 	
			input: tval = trg_dwn_chk_val
			output: dden = trg_dwn_equ_den
			algorithm: 	
				tmpDen = trg_dwn_equ_den[parent] * sclvec
				trg_dwn_chk_val[i] += dgemv (DE2DC, tmpDen)
				tmpDen = dgemv (DC2DE, tval[i])
				trg_dwn_equ_den[i] = tmpDen * sclvec
		 */	

		down_eval__gpu_non_leaves <<<GB, TB>>> (nodeVec.size (),
																						i,
																						f->tag_d_,
																						f->depth_d_,
																						f->path2Node_d_,
																						f->parent_d_,
																						f->SP_DC_n_padded_,
																						f->TRG_DWN_CHK_VAL_d_,
																						f->DC2DE_c,
																						f->DC2DE_r,
																						f->DC2DE_r_padded,
																						f->DC2DE_d_,
																						f->DE2DC_c,
																						f->DE2DC_r,
																						f->DE2DC_r_padded,
																						f->DE2DC_d_,
																						f->SP_DE_n_padded,
																						f->TRG_DWN_EQU_DEN_d_);
	  cudaThreadSynchronize ();
	  gpu_check_error (stderr);

	}

	/* do leaves */
	down_eval__gpu_leaves <<<GB, TB>>> (nodeVec.size (),
																			f->T_d_.Bptr_,
																			f->T_d_.Bn_,
																			f->T_d_.x_,
																			f->T_d_.y_,
																			f->T_d_.z_,
																			f->T_d_.w_,
																			f->SP_DE_n_,
																			f->SP_DE_n_padded,
																			f->SP_DE_d_,
																			f->TRG_DWN_EQU_DEN_d_,
																			f->radius_d_,
																			f->center0_d_,
																			f->center1_d_,
																			f->center2_d_
																			);
	cudaThreadSynchronize ();
	gpu_check_error (stderr);

  return 0;
}

__global__
void
wlist_eval__gpu (int n_boxes_,
								 int *tag_,
								 int *srcNum_,
								 int *child_,
								 int *Bptr_T_,
								 int *Bn_T_,
								 dtype *x_T_,
								 dtype *y_T_,
								 dtype *z_T_,
								 dtype *w_T_,
								 int *Bptr_S_,
								 int *Bn_S_,
								 dtype *x_S_,
								 dtype *y_S_,
								 dtype *z_S_,
								 dtype *w_S_,
								 int *L__w,
								 int *Ptr__w,
								 int sp_ue_n_,
								 int sp_ue_n_padded,
								 dtype *sp_ue_,
								 dtype *radius_,
								 dtype *center0_,
								 dtype *center1_,
								 dtype *center2_,
								 dtype *src_upw_equ_den_
								)
{
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

	int wn, k, t, s;

  __shared__ dtype SL_POS[4][SL_POS_SIZE];

	/* valid box */
	if(bid < n_boxes_) {
		/* if target box */
		if(tag_[bid] & LET_TRGNODE) {
			/* if target box is a leaf */
			if(child_[bid] == -1) {
				/* what are this box's target points */
        int trg_begin = Bptr_T_[bid];
        int trg_end = Bptr_T_[bid] + Bn_T_[bid];

				/* what are this box's W list */
        int w_start = Ptr__w[bid];
        int w_end = Ptr__w[bid + 1];

				/* do this only if box has points AND a list of W nodes */
				if(trg_begin < trg_end && w_start < w_end) {
	
					/* for each w nodes in this box's list */
					for(wn = w_start; wn < w_end; wn++) {
						/* node id */
						int src = L__w[wn];

						/* if conditions are met, compute localpos */
						if(child_[src] != -1 || srcNum_[src] >= sp_ue_n_) {
							dtype c0 = center0_[src];
							dtype c1 = center1_[src];
							dtype c2 = center2_[src];
							dtype r = radius_[src];
							for(k = tid; k < sp_ue_n_padded; k += blockDim.x) {
								SL_POS[0][k] = c0 + r * sp_ue_[k];
								SL_POS[1][k] = c1 + r * sp_ue_[sp_ue_n_padded + k];
								SL_POS[2][k] = c2 + r * sp_ue_[2 * sp_ue_n_padded + k];
								SL_POS[3][k] = src_upw_equ_den_[src * sp_ue_n_padded + k];
							}
						}
						__syncthreads ();

						if(child_[src] == -1 && srcNum_[src] < sp_ue_n_) {
							/* go through each target point */
							for(t = trg_begin + tid; t < trg_end; t += blockDim.x) {
								dtype xt = x_T_[t];
								dtype yt = y_T_[t];
								dtype zt = z_T_[t];
								dtype wt = 0.0;
				
								/* go through each source point */
								int src_begin = Bptr_S_[src];
								int src_end = Bptr_S_[src] + Bn_S_[src];
								for(s = src_begin; s < src_end; s++) {
									dtype xs = xt - x_S_[s];
									dtype ys = yt - y_S_[s];
									dtype zs = zt - z_S_[s];
					
									dtype rsq = xs * xs + ys * ys + zs * zs;
									rsq = rsqrt (rsq);

									wt += w_S_[s] * rsq;
								}

								w_T_[t] += wt * OOFP_R;
							}
						} else {
							/* go through each target point */
							for(t = trg_begin + tid; t < trg_end; t += blockDim.x) {
								dtype xt = x_T_[t];
								dtype yt = y_T_[t];
								dtype zt = z_T_[t];
								dtype wt = 0.0;

								/* go through each sl_pos point */
								for(s = 0; s < sp_ue_n_; s++) {
									dtype xs = xt - SL_POS[0][s];
									dtype ys = yt - SL_POS[1][s];
									dtype zs = zt - SL_POS[2][s];

									dtype rsq = xs * xs + ys * ys + zs * zs;
									rsq = rsqrt (rsq);

									wt += SL_POS[3][s] * rsq;
								}
						
								w_T_[t] += wt * OOFP_R;
							}
						}
						__syncthreads ();
					}
				}
			}
		}
	}
}


__global__
void
wlist_eval__gpu_ (int n_boxes_,
								 int *tag_,
								 int *srcNum_,
								 int *child_,
								 int *Bptr_T_,
								 int *Bn_T_,
								 dtype *x_T_,
								 dtype *y_T_,
								 dtype *z_T_,
								 dtype *w_T_,
								 int *Bptr_S_,
								 int *Bn_S_,
								 dtype *x_S_,
								 dtype *y_S_,
								 dtype *z_S_,
								 dtype *w_S_,
								 int *L__w,
								 int *Ptr__w,
								 int sp_ue_n_,
								 int sp_ue_n_padded,
								 dtype *sp_ue_,
								 dtype *radius_,
								 dtype *center0_,
								 dtype *center1_,
								 dtype *center2_,
								 int uc2ue_r,
								 int uc2ue_r_padded,
								 dtype *src_upw_equ_den_
								)
{
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  int i, j, k;

  __shared__ dtype SL_POS[4][SL_POS_SIZE];

	/* valid box */
	if(bid < n_boxes_) {
		/* if target box */
		if(tag_[bid] & LET_TRGNODE) {
			/* if target box is a leaf */
			if(child_[bid] == -1) {

				/* loop boundaries */
				int w_start = Ptr__w[bid];
				int w_end = Ptr__w[bid + 1];
				int trg_begin = Bptr_T_[bid];
				int trg_end = Bptr_T_[bid] + Bn_T_[bid];

				/* start only if all loop boundaries have actual work */
				if(w_start < w_end && trg_begin < trg_end) {

					/* each thread gets 1 source */
					for(i = trg_begin + tid; i < trg_end; i += blockDim.x) {
						dtype xt = x_T_[i];
						dtype yt = y_T_[i];
						dtype zt = z_T_[i];

						/* for each target point loop over wlist */
						for(j = w_start; j < w_end; j++) {
							dtype wt = 0.0;
							int src = L__w[j];

							if(child_[src] != -1 || srcNum_[src] >= sp_ue_n_) {
								dtype c0 = center0_[src];
								dtype c1 = center1_[src];
								dtype c2 = center2_[src];
								dtype r = radius_[src];
								for(k = tid; k < sp_ue_n_padded; k += blockDim.x) {
									SL_POS[0][k] = c0 + r * sp_ue_[k];
									SL_POS[1][k] = c1 + r * sp_ue_[sp_ue_n_padded + k];
									SL_POS[2][k] = c2 + r * sp_ue_[2 * sp_ue_n_padded + k];
									SL_POS[3][k] = src_upw_equ_den_[bid * sp_ue_n_padded + k];
								}	
								__syncthreads ();
							}


							if(child_[src] == -1 && srcNum_[src] < sp_ue_n_) {
								// if(bid == 31 && tid == 0) printf("==> %d\n", src);
								int src_begin = Bptr_S_[src];
								int src_end = Bptr_S_[src] + Bn_S_[src];
								/* loop over points in the source box */
								for(k = src_begin; k < src_end; k++) {
									dtype xs = x_S_[k];
									dtype ys = y_S_[k];
									dtype zs = z_S_[k];
									// dtype ws = w_S_[k];

									xs = xt - xs;
									ys = yt - ys;
									zs = zt - zs;

									dtype rsq = xs * xs + ys * ys + zs * zs;
									rsq = rsqrt (rsq);
			
									wt += wt * rsq;
								}
								// wt = wt * OOFP_R;
								w_T_[i] += wt * OOFP_R;
							} else {
							
								/* ulist_calc */
								for(k = 0; k < sp_ue_n_; k++) {
									dtype x = xt - SL_POS[0][k];
									dtype y = yt - SL_POS[1][k];
									dtype z = zt - SL_POS[2][k];
								
									dtype rsq = x * x + y * y + z * z;
									rsq = rsqrt (rsq);
					
									wt += SL_POS[3][k] * rsq;
								}
								// wt = wt * OOFP_R;
								w_T_[i] += wt * OOFP_R;
							} /* if src == -1 && srcNum < SP[UE].n */
						} /* for each wlist neighbor */

						/* write result back */
						// w_T_[i] += wt;
					} /* for each target point */
				} /* if there is work to do */
			}
		}
	}

}

int
wlist_calc__gpu(FMMWrapper_t *f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* do level 0~4 
   * nothing is done for level 0 and 1 */
  const int NB = get_thread_block_size_down ();
  const int NG = nodeVec.size ();

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

	wlist_eval__gpu <<<GB, TB>>> (nodeVec.size (),
																f->tag_d_,
																f->srcNum_d_,
																f->child_d_,
																f->T_d_.Bptr_,
																f->T_d_.Bn_,
																f->T_d_.x_,
																f->T_d_.y_,
																f->T_d_.z_,
																f->T_d_.w_,
																f->S_d_.Bptr_,
																f->S_d_.Bn_,
																f->S_d_.x_,
																f->S_d_.y_,
																f->S_d_.z_,
																f->S_d_.w_,
																f->W_d_.L_,
																f->W_d_.Ptr_,
																f->SP_UE_n_,
																f->SP_UE_n_padded,
																f->SP_UE_d_,
                               	f->radius_d_,
                               	f->center0_d_,
                               	f->center1_d_,
                               	f->center2_d_,
																f->SRC_UPW_EQU_DEN_d_
																);
	cudaThreadSynchronize ();
	gpu_check_error (stderr);


	return 0;
}


__global__
void
xlist_eval__gpu (int n_boxes_,
								 int *tag_,
								 int *trgNum_,
								 int *child_,
								 int *Bptr_T_,
								 int *Bn_T_,
								 dtype *x_T_,
								 dtype *y_T_,
								 dtype *z_T_,
								 dtype *w_T_,
								 int *Bptr_S_,
								 int *Bn_S_,
								 dtype *x_S_,
								 dtype *y_S_,
								 dtype *z_S_,
								 dtype *w_S_,
								 int *L__x,
								 int *Ptr__x,
								 int sp_dc_n_,
								 int sp_dc_n_padded,
								 dtype *sp_dc_,
								 dtype *radius_,
								 dtype *center0_,
								 dtype *center1_,
								 dtype *center2_,
								 dtype *trg_dwn_chk_val_
								)
{
	int tid = threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;

	int xn, k, t, s;

	__shared__ dtype SL_POS[4][SL_POS_SIZE];

	if(bid < n_boxes_) {
		if(tag_[bid] & LET_TRGNODE) {
			int x_start = Ptr__x[bid];
			int x_end = Ptr__x[bid + 1];
			/* do work if there is a x list */
			if(x_start < x_end) {

				/* for each node in x list */
				for(xn = x_start; xn < x_end; xn++) {
					int src = L__x[xn];

					if(child_[bid] != -1 || trgNum_[bid] >= sp_dc_n_) {
						dtype c0 = center0_[bid];
						dtype c1 = center1_[bid];
						dtype c2 = center2_[bid];
						dtype r = radius_[bid];
						for(k = tid; k < sp_dc_n_padded; k += blockDim.x) {
							SL_POS[0][k] = c0 + r * sp_dc_[k];
							SL_POS[1][k] = c1 + r * sp_dc_[sp_dc_n_padded + k];
							SL_POS[2][k] = c2 + r * sp_dc_[2 * sp_dc_n_padded + k];
							SL_POS[3][k] = trg_dwn_chk_val_[bid * sp_dc_n_padded + k];
						}
					}
					__syncthreads ();

					if(child_[bid] == -1 && trgNum_[bid] < sp_dc_n_) {
						int trg_begin = Bptr_T_[bid];
						int trg_end = Bptr_T_[bid] + Bn_T_[bid];
						for(t = trg_begin + tid; t < trg_end; t += blockDim.x) {
							dtype xt = x_T_[t];
							dtype yt = y_T_[t];
							dtype zt = z_T_[t];
							dtype wt = 0.0;

							int src_begin = Bptr_S_[src];
							int src_end = Bptr_S_[src] + Bn_S_[src];
							for(s = src_begin; s < src_end; s++) {
								dtype xs = xt - x_S_[s];
								dtype ys = yt - y_S_[s];
								dtype zs = zt - z_S_[s];

								dtype rsq = xs * xs + ys * ys + zs * zs;
								rsq = rsqrt (rsq);

								wt += w_S_[s] * rsq;
							}
							w_T_[t] += wt * OOFP_R;
						}
					} else {
						for(t = tid; t < sp_dc_n_; t += blockDim.x) {
							dtype xt = SL_POS[0][t];
							dtype yt = SL_POS[1][t];
							dtype zt = SL_POS[2][t];
							dtype wt = 0.0;

							int src_begin = Bptr_S_[src];
							int src_end = Bptr_S_[src] + Bn_S_[src];
							for(s = src_begin;  s < src_end; s++) {
								dtype xs = xt - x_S_[s];
								dtype ys = yt - y_S_[s];
								dtype zs = zt - z_S_[s];
			
								dtype rsq = xs * xs + ys * ys + zs * zs;
								rsq = rsqrt (rsq);

								wt += w_S_[s] * rsq;
							}
							trg_dwn_chk_val_[bid * sp_dc_n_padded + t] = SL_POS[3][t] + 
																													 wt * OOFP_R;
						}
					}
					__syncthreads ();
				}
			} /* if xlist is not 0 */
		} /* if box is target node */
	} /* if bid is withint nodeVec.size () */
}

int
xlist_calc__gpu(FMMWrapper_t *f)
{

  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* do level 0~4 
   * nothing is done for level 0 and 1 */
  const int NB = get_thread_block_size_down ();
  const int NG = nodeVec.size ();

  dim3 GB (65535, (NG / 65535) + 1, 1);
  dim3 TB (NB, 1, 1);

	xlist_eval__gpu <<<GB, TB>>> (nodeVec.size (),
																f->tag_d_,
																f->trgNum_d_,
																f->child_d_,
																f->T_d_.Bptr_,
																f->T_d_.Bn_,
																f->T_d_.x_,
																f->T_d_.y_,
																f->T_d_.z_,
																f->T_d_.w_,
																f->S_d_.Bptr_,
																f->S_d_.Bn_,
																f->S_d_.x_,
																f->S_d_.y_,
																f->S_d_.z_,
																f->S_d_.w_,
																f->X_d_.L_,
																f->X_d_.Ptr_,
																f->SP_DC_n_,
																f->SP_DC_n_padded_,
																f->SP_DC_d_,
                               	f->radius_d_,
                               	f->center0_d_,
                               	f->center1_d_,
                               	f->center2_d_,
																f->TRG_DWN_CHK_VAL_d_
																);
	cudaThreadSynchronize ();
	gpu_check_error (stderr);

	return 0;
}

int
copy_trg_val__gpu (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;
  /* transfer data back */
  cudaMemcpy (f->T_h_.w_, f->T_d_.w_, f->T_h_.Bptr_[f->T_h_.n_boxes_] * 
      sizeof (dtype), cudaMemcpyDeviceToHost);

  /* convert gpu results to cpu */
  for(int i = 0; i < f->T_h_.n_boxes_; i++) {
    for(int j = f->T_h_.Bptr_[i]; j < f->T_h_.Bptr_[i + 1]; j++) {
      All_N->Nt[i].den_pot[j - f->T_h_.Bptr_[i]] += f->T_h_.w_[j];
    }
  }
  for (int i = 0; i < nodeVec.size(); i++) 
	  if( nodeVec[i].tag & LET_TRGNODE)  
      if (nodeVec[i].child == -1) {
        set_value (nodeVec[i].trgNum, All_N->pot_orig, All_N->Nt[i].den_pot, nodeVec[i].trgOwnVecIdxs);
		  }

  return 0;
}

/* eof */
