#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "evaluate.h"
#include "util.h"
#include "reals.h"
#include "partial.h"
#include "../timing/timing.h"
#include "node_gpu.h"
#include <cutil_inline.h>

/* ------------------------------------------------------------------------
 */
int
get_byte_padding()
{
	return getenv__int("BYTEPAD", 128);
}

/* ------------------------------------------------------------------------
 */
void
xlist_create_xlist (UList_t* U, int num_boxes, AllNodes* All_N)
{
	int i, j, nu;
	int list_size = 0;
	
	assert (U && All_N);

	vector<NodeTree>& nodeVec = *All_N->N;

	/* allocate memory for ulist ptr */
	U->n_boxes_ = num_boxes;
	U->Ptr_ = (int *) malloc (sizeof (int) * (num_boxes + 1));
	assert (U->Ptr_);

	/* See how big ulist should be */
	U->Ptr_[0] = 0;
	for(i = 0; i < U->n_boxes_; i++) {
		list_size += nodeVec[i].Xnodes.size ();
		U->Ptr_[i + 1] = list_size;
	}

	/* allocate memory for ulist */
	U->L_ = (int*) malloc (sizeof (int) * list_size);
	assert (U->L_);

	/* initialize ulist */
	for(i = 0; i < U->n_boxes_; i++) {
		nu = nodeVec[i].Xnodes.size ();
		for(j = 0; j < nu; j++) {
			U->L_[U->Ptr_[i] + j] = nodeVec[i].Xnodes[j];
		}
	}
}




void
wlist_create_wlist (UList_t* U, int num_boxes, AllNodes* All_N)
{
	int i, j, nu;
	int list_size = 0;
	
	assert (U && All_N);

	vector<NodeTree>& nodeVec = *All_N->N;

	/* allocate memory for ulist ptr */
	U->n_boxes_ = num_boxes;
	U->Ptr_ = (int *) malloc (sizeof (int) * (num_boxes + 1));
	assert (U->Ptr_);

	/* See how big ulist should be */
	U->Ptr_[0] = 0;
	for(i = 0; i < U->n_boxes_; i++) {
		list_size += nodeVec[i].Wnodes.size ();
		U->Ptr_[i + 1] = list_size;
	}

	/* allocate memory for ulist */
	U->L_ = (int*) malloc (sizeof (int) * list_size);
	assert (U->L_);

	/* initialize ulist */
	for(i = 0; i < U->n_boxes_; i++) {
		nu = nodeVec[i].Wnodes.size ();
		for(j = 0; j < nu; j++) {
			U->L_[U->Ptr_[i] + j] = nodeVec[i].Wnodes[j];
		}
	}
}


void
ulist_create_ulist (UList_t* U, int num_boxes, AllNodes* All_N)
{
	int i, j, nu;
	int list_size = 0;
	
	assert (U && All_N);

	vector<NodeTree>& nodeVec = *All_N->N;

	/* allocate memory for ulist ptr */
	U->n_boxes_ = num_boxes;
	U->Ptr_ = (int *) malloc (sizeof (int) * (num_boxes + 1));
	assert (U->Ptr_);

	/* See how big ulist should be */
	U->Ptr_[0] = 0;
	for(i = 0; i < U->n_boxes_; i++) {
		list_size += nodeVec[i].Unodes.size ();
		U->Ptr_[i + 1] = list_size;
	}

	/* allocate memory for ulist */
	U->L_ = (int*) malloc (sizeof (int) * list_size);
	assert (U->L_);

	/* initialize ulist */
	for(i = 0; i < U->n_boxes_; i++) {
		nu = nodeVec[i].Unodes.size ();
		for(j = 0; j < nu; j++) {
			U->L_[U->Ptr_[i] + j] = nodeVec[i].Unodes[j];
		}
	}
}

/* ------------------------------------------------------------------------
 */

void
ulist_create_boxes__double_source (AllNodes *All_N, FMMWrapper_t *F)
{
	int i, j, n;
	int padding, n_padded, n_points_, n_points_padded_;

	vector<NodeTree>& nodeVec = *All_N->N;

	Boxes_t *B;
	Node *N;

	B = &F->S_h_;
	N = All_N->Ns;

	assert (B && N);

	padding = get_byte_padding () / sizeof (dtype);

	B->n_boxes_ = nodeVec.size ();
	B->Bptr_ = (int *) malloc (sizeof (int) * (B->n_boxes_ + 1));
	B->Bn_ = (int *) malloc (sizeof (int) * B->n_boxes_);
	assert (B->Bptr_ && B->Bn_);

	n_points_ = 0;
	n_points_padded_ = 0;
	B->Bptr_[0] = 0;
	
	for(i = 0; i < B->n_boxes_; i++) {
		if(nodeVec[i].tag & LET_SRCNODE && nodeVec[i].child == -1) {
			n = N[i].num_pts;
			n_padded = ((n + padding - 1) / padding) * padding;
			assert (n_padded >= n);

			B->Bn_[i] = n;
			B->Bptr_[i + 1] = B->Bptr_[i] + n_padded;

			n_points_ += n;
			n_points_padded_ += n_padded;
		} else {
			B->Bn_[i] = 0;
			B->Bptr_[i + 1] = B->Bptr_[i];
		}
	}
	assert (n_points_padded_ == B->Bptr_[B->n_boxes_]);

	B->n_points_ = n_points_;
	B->x_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	B->y_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	B->z_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	B->w_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	assert (B->x_ && B->y_ && B->z_ && B->w_);

	/* copy points */
	for(i = 0; i < B->n_boxes_; i++) {
		if(nodeVec[i].tag & LET_SRCNODE) {
			n = N[i].num_pts;
			for(j = 0; j < n; j++) {
				B->x_[B->Bptr_[i] + j] = N[i].x[j];
				B->y_[B->Bptr_[i] + j] = N[i].y[j];
				B->z_[B->Bptr_[i] + j] = N[i].z[j];
				B->w_[B->Bptr_[i] + j] = N[i].den_pot[j];
			}
		}
	}

}


void
ulist_create_boxes__double_target (AllNodes *All_N, FMMWrapper_t *F)
{
	int i, j, n;
	int padding, n_padded, n_points_, n_points_padded_;

	vector<NodeTree>& nodeVec = *All_N->N;

	Boxes_t *B;
	Node *N;

	B = &F->T_h_;
	N = All_N->Nt;

	assert (B && N);

	padding = get_byte_padding () / sizeof (dtype);

	B->n_boxes_ = nodeVec.size ();
	B->Bptr_ = (int *) malloc (sizeof (int) * (B->n_boxes_ + 1));
	B->Bn_ = (int *) malloc (sizeof (int) * B->n_boxes_);
	assert (B->Bptr_ && B->Bn_);

	n_points_ = 0;
	n_points_padded_ = 0;
	B->Bptr_[0] = 0;
	
	for(i = 0; i < B->n_boxes_; i++) {
		if(nodeVec[i].tag & LET_TRGNODE && nodeVec[i].child == -1) {
			n = N[i].num_pts;
			n_padded = ((n + padding - 1) / padding) * padding;
			assert (n_padded >= n);

			B->Bn_[i] = n;
			B->Bptr_[i + 1] = B->Bptr_[i] + n_padded;

			n_points_ += n;
			n_points_padded_ += n_padded;
		} else {
			B->Bn_[i] = 0;
			B->Bptr_[i + 1] = B->Bptr_[i];
		}
	}
	assert (n_points_padded_ == B->Bptr_[B->n_boxes_]);

	B->n_points_ = n_points_;
	B->x_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	B->y_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	B->z_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	B->w_ = (real_t *) malloc (n_points_padded_ * sizeof (real_t));
	assert (B->x_ && B->y_ && B->z_ && B->w_);

	/* copy points */
	for(i = 0; i < B->n_boxes_; i++) {
		if(nodeVec[i].tag & LET_TRGNODE) {
			n = N[i].num_pts;
			for(j = 0; j < n; j++) {
				B->x_[B->Bptr_[i] + j] = N[i].x[j];
				B->y_[B->Bptr_[i] + j] = N[i].y[j];
				B->z_[B->Bptr_[i] + j] = N[i].z[j];
				B->w_[B->Bptr_[i] + j] = N[i].den_pot[j];
			}
		}
	}
}

void
ulist_create_boxes__double (Boxes_t* B, 
													 int num_boxes, 
													 const Node* N, 
													 int padding)
{
	int i, j, n, n_padded, min;

	/* total number of points */
	int n_points_ = 0;
	int n_points_padded_ = 0;

	/* check if structures that were passed in are valid */
	assert (B && N);

	/* allocate memory to data structures */
	B->n_boxes_ = num_boxes;
	B->Bptr_ = (int *) malloc (sizeof (int) * (B->n_boxes_ + 1));
	B->Bn_ = (int *) malloc (sizeof (int) * B->n_boxes_);
	assert (B->Bptr_ && B->Bn_);

	/* initialize data structures */
	B->Bptr_[0] = 0;
	for(i = 0; i < num_boxes; i++) {
		/* number of points in this box */
		n = N[i].num_pts;
		/* number of points in this box if padded */
		n_padded = ((n + padding - 1) / padding) * padding;
		assert (n_padded >= n);

		/* make Bn_ and Bptr_ have/point to the right values */
		B->Bn_[i] = n;
		B->Bptr_[i+1] = B->Bptr_[i] + n_padded;

		n_points_ += n;
		n_points_padded_ += n_padded;
	}
	assert (n_points_padded_ == B->Bptr_[B->n_boxes_]);

	/* allocate memory to data structures that are going to hold the values */
	B->n_points_ = n_points_;	
	B->x_ = (real_t*) malloc (n_points_padded_ * sizeof (real_t));
	B->y_ = (real_t*) malloc (n_points_padded_ * sizeof (real_t));
	B->z_ = (real_t*) malloc (n_points_padded_ * sizeof (real_t));
	B->w_ = (real_t*) malloc (n_points_padded_ * sizeof (real_t));
	assert (B->x_ && B->y_ && B->z_ && B->w_);

	/* copy points */
	for(i = 0; i < num_boxes; i++) {
		n = N[i].num_pts;
		min = B->Bptr_[i];
		for(j = 0; j < n; j++) {
			B->x_[min + j] = N[i].x[j];
			B->y_[min + j] = N[i].y[j];
			B->z_[min + j] = N[i].z[j];
			B->w_[min + j] = N[i].den_pot[j];
		}
	}
}

/* ------------------------------------------------------------------------
 */

void
alloc__SOURCE_BOX__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* Source boxes */
	f->S_d_.n_points_ = f->S_h_.n_points_;
	f->S_d_.n_boxes_ = f->S_h_.n_boxes_;
	/* Allocate memory for data */
	cutilSafeCall (cudaMalloc ((void**)&f->S_d_.x_, 
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] 
														 * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->S_d_.y_, 
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] 
														 * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->S_d_.z_, 
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] 
														 * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->S_d_.w_, 
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] 
														 * sizeof (dtype)));
	/* Allocate memory for pointers */
	cutilSafeCall (cudaMalloc ((void**)&f->S_d_.Bptr_,
														 (f->S_d_.n_boxes_ + 1) * sizeof (int)));
	cutilSafeCall (cudaMalloc ((void**)&f->S_d_.Bn_,
														 f->S_d_.n_boxes_ * sizeof (int)));
	assert (&f->S_d_ && &f->S_h_);
  /* ------------------------------------------------------------ */
}

void
alloc__TARGET_BOX__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* Target boxes */
	f->T_d_.n_points_ = f->T_h_.n_points_;
	f->T_d_.n_boxes_ = f->T_h_.n_boxes_;
	
	/* Allocate memory for data */
	cutilSafeCall (cudaMalloc ((void**)&f->T_d_.x_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] 
														 * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->T_d_.y_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] 
														 * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->T_d_.z_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] 
														 * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->T_d_.w_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] 
														 * sizeof (dtype)));
	/* Allocate memory for pointers */
	cutilSafeCall (cudaMalloc ((void**)&f->T_d_.Bptr_,
														 (f->T_h_.n_boxes_ + 1) * sizeof (int)));
	cutilSafeCall (cudaMalloc ((void**)&f->T_d_.Bn_,
														 f->T_h_.n_boxes_ * sizeof (int)));
	assert (&f->T_d_ && &f->T_h_);
  /* ------------------------------------------------------------ */
}

void
alloc__U_LIST__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* ulist */
	f->U_d_.n_boxes_ = f->U_h_.n_boxes_;
	cutilSafeCall (cudaMalloc ((void**)&f->U_d_.L_,
														 f->U_h_.Ptr_[f->U_h_.n_boxes_] * sizeof (int)));
	cutilSafeCall (cudaMalloc ((void**)&f->U_d_.Ptr_,
														 (f->U_h_.n_boxes_ + 1) * sizeof (int)));
	assert (&f->U_d_ && &f->U_h_);
  /* ------------------------------------------------------------ */
}

void
alloc__TAG__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* tag */
	cutilSafeCall (cudaMalloc ((void**)&f->tag_d_, 
														 nodeVec.size () * sizeof (int)));
	assert (f->tag_d_);
  /* ------------------------------------------------------------ */
}


void
alloc__DEPTH__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* depth */
	cutilSafeCall (cudaMalloc ((void**)&f->depth_d_, 
														 nodeVec.size () * sizeof (int)));
	assert (f->depth_d_);
  /* ------------------------------------------------------------ */
}


void
alloc__CHILDREN__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* children */
	/*
	cutilSafeCall (cudaMalloc ((void**)&f->child_d_, 
														 num_non_leaf_nodes * sizeof (int)));
	 */
	cutilSafeCall (cudaMalloc ((void**)&f->child_d_, 
														 nodeVec.size () * sizeof (int)));
	assert (f->child_d_);
  /* ------------------------------------------------------------ */
}


void
alloc__RADIUS__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* radius */
	/*
	cutilSafeCall (cudaMalloc ((void**)&f->radius_d_,
														 num_leaf_nodes * sizeof (dtype)));
	 */
	cutilSafeCall (cudaMalloc ((void**)&f->radius_d_,
														 nodeVec.size () * sizeof (dtype)));
	assert (f->radius_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__CENTER__ (FMMWrapper_t* f) {
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* radius */
	/*
	cutilSafeCall (cudaMalloc ((void**)&f->radius_d_,
														 num_leaf_nodes * sizeof (dtype)));
	 */
	cutilSafeCall (cudaMalloc ((void**)&f->radius_d_,
														 nodeVec.size () * sizeof (dtype)));
  /* ------------------------------------------------------------ */
	/* center */
	cutilSafeCall (cudaMalloc ((void**)&f->center0_d_,
														 nodeVec.size () * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->center1_d_,
														 nodeVec.size () * sizeof (dtype)));
	cutilSafeCall (cudaMalloc ((void**)&f->center2_d_,
														 nodeVec.size () * sizeof (dtype)));
	assert (f->center0_d_ && f->center1_d_ && f->center2_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__SP_UC__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* SP[UC] */
	cutilSafeCall (cudaMalloc ((void**)&f->SP_UC_d_,
														 3 * f->SP_UC_size_padded * sizeof (dtype)));
	assert (f->SP_UC_d_);
  /* ------------------------------------------------------------ */
}


void
alloc__UC2UE__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* UC2UE matrix */
	cutilSafeCall (cudaMalloc ((void**)&f->UC2UE_d_,
														 f->UC2UE_r_padded * f->UC2UE_c * sizeof (dtype)));
	assert (f->UC2UE_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__UE2UC__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* UE2UC matrix */
	cutilSafeCall (cudaMalloc ((void**)&f->UE2UC_d_,
														 (2 * 2 * 2) * (f->UE2UC_r_padded * f->UE2UC_c) *
														 sizeof (dtype)));
	assert (f->UE2UC_d_);
  /* ------------------------------------------------------------ */
}


void
alloc__SRC_UPW_EQU_DEN__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* Temporary up_calc GPU variables */
	/* src_upw_equ_den */
	cutilSafeCall (cudaMalloc ((void**)&f->SRC_UPW_EQU_DEN_d_,
														 nodeVec.size () * f->UC2UE_r_padded * 
														 sizeof (dtype)));
	assert (f->SRC_UPW_EQU_DEN_d_);
  /* ------------------------------------------------------------ */

}


void
alloc__VLIST_SRC__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* src */
	cutilSafeCall (cudaMalloc ((void**)&f->vlist_src_d_,
														 nodeVec.size () * f->vlist_array_size *
														 sizeof (dtype)));
	assert (f->vlist_src_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__REG_DEN__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
  /* reg_den */
	cutilSafeCall (cudaMalloc ((void**)&f->reg_den_d_,
														 nodeVec.size () * f->reg_den_size *
														 sizeof (dtype)));
	assert (f->reg_den_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__TT__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* tt */
	cutilSafeCall (cudaMalloc ((void**)&f->tt, f->trans_arrays_num * f->RP_n_ * 	
														 sizeof (dtype)));
	assert (f->tt);
  /* ------------------------------------------------------------ */
}

void
alloc__VLIST_TRANS__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
  /* trans */
	cutilSafeCall (cudaMalloc ((void**)&f->vlist_trans_d_,
														 f->trans_arrays_num * f->vlist_array_size *
														 sizeof (dtype)));
  assert (f->vlist_trans_d_);
  /* ------------------------------------------------------------ */
}


void
alloc__VLIST_TRG__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* trg */
	cutilSafeCall (cudaMalloc ((void**)&f->vlist_trg_d_,
														 nodeVec.size () * f->vlist_array_size * 
														 sizeof (dtype)));
	assert (f->vlist_trg_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__VLIST_TLIST__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* vlist and tlist and pointer */
	/* pointer */
	cutilSafeCall (cudaMalloc ((void**)&f->vlist_ptr_d_, 
														 (nodeVec.size () + 1) * sizeof (int)));
	assert (f->vlist_ptr_d_);
	/* vlist */
	cutilSafeCall (cudaMalloc ((void**)&f->vlist_d_, f->list_size * sizeof (int)));
	assert (f->vlist_d_);
	/* tlist */
	cutilSafeCall (cudaMalloc ((void**)&f->tlist_d_, f->list_size * sizeof (int)));
	assert (f->tlist_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__TRG_DWN_CHK_VAL__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
  /* trg_dwn_chk_val */
	cutilSafeCall (cudaMalloc ((void**)&f->TRG_DWN_CHK_VAL_d_,
														 nodeVec.size () * f->SP_DC_n_padded_ * 
														 sizeof (dtype)));
	assert (f->TRG_DWN_CHK_VAL_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__PATH2NODE__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* path2Node */
	cutilSafeCall (cudaMalloc ((void**)&f->path2Node_d_,
														 nodeVec.size () * sizeof (int3)));
	assert (f->path2Node_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__PARENT__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* parent */
	/* Not needed - use children structure instead */
	/* Now I need it */
	cutilSafeCall (cudaMalloc ((void**)&f->parent_d_, 
														 nodeVec.size () * sizeof (int)));
	assert (f->parent_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__TRG_DWN_EQU_DEN__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* ------------------------------------------------------------ */
	/* trg_dwn_equ_den */
	cutilSafeCall (cudaMalloc ((void**)&f->TRG_DWN_EQU_DEN_d_,
														 nodeVec.size () * f->SP_DE_n_padded * 
														 sizeof (dtype)));
	assert (f->TRG_DWN_EQU_DEN_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__DC2DE__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* DC2DE_mat */
	cutilSafeCall (cudaMalloc ((void**)&f->DC2DE_d_,
														 f->DC2DE_r_padded * f->DC2DE_c * sizeof (dtype)));
	assert (f->DC2DE_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__DE2DC__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* DE2DC_mat[8] */
	cutilSafeCall (cudaMalloc ((void**)&f->DE2DC_d_,
														 (2 * 2 * 2) * f->DE2DC_r_padded * f->DE2DC_c *
														 sizeof (dtype)));
	assert (f->DE2DC_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__SP_DE__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* SP[DE] */
	cutilSafeCall (cudaMalloc ((void**)&f->SP_DE_d_,
														 3 * f->SP_DE_n_padded * sizeof (dtype)));
	assert (f->SP_DE_d_);
  /* ------------------------------------------------------------ */
}

void
alloc__SP_UE__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMalloc ((void**)&f->SP_UE_d_,
														 3 * f->SP_UE_n_padded * sizeof (dtype)));
	assert (f->SP_UE_d_);
}

void
alloc__W_LIST__ (FMMWrapper_t* f) 
{
  /* ------------------------------------------------------------ */
	/* ulist */
	f->W_d_.n_boxes_ = f->W_h_.n_boxes_;
	cutilSafeCall (cudaMalloc ((void**)&f->W_d_.L_,
														 f->W_h_.Ptr_[f->W_h_.n_boxes_] * sizeof (int)));
	cutilSafeCall (cudaMalloc ((void**)&f->W_d_.Ptr_,
														 (f->W_h_.n_boxes_ + 1) * sizeof (int)));
	assert (&f->W_d_ && &f->W_h_);
  /* ------------------------------------------------------------ */
}

void
alloc__SRCNUM__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMalloc ((void**)&f->srcNum_d_,
														 nodeVec.size () * sizeof (int)));
	assert (f->srcNum_d_);
}

void
alloc__SP_DC__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMalloc ((void**)&f->SP_DC_d_,
														 3 * f->SP_DC_n_padded_ * sizeof (dtype)));
	assert (f->SP_DC_d_);
}

void
alloc__X_LIST__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* ulist */
	f->X_d_.n_boxes_ = f->X_h_.n_boxes_;
	cutilSafeCall (cudaMalloc ((void**)&f->X_d_.L_,
														 f->X_h_.Ptr_[f->X_h_.n_boxes_] * sizeof (int)));
	cutilSafeCall (cudaMalloc ((void**)&f->X_d_.Ptr_,
														 (f->X_h_.n_boxes_ + 1) * sizeof (int)));
	assert (&f->X_d_ && &f->X_h_);
  /* ------------------------------------------------------------ */
}

void
alloc__TRGNUM__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMalloc ((void**)&f->trgNum_d_,
														 nodeVec.size () * sizeof (int)));
	assert (f->trgNum_d_);
}


void
xfer__SOURCE_BOX__ (FMMWrapper_t* f)
{
	/* Source boxes */
	cutilSafeCall (cudaMemcpy (f->S_d_.x_, f->S_h_.x_,
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->S_d_.y_, f->S_h_.y_,
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->S_d_.z_, f->S_h_.z_,
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->S_d_.w_, f->S_h_.w_,
														 f->S_h_.Bptr_[f->S_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->S_d_.Bptr_, f->S_h_.Bptr_,
														 (f->S_h_.n_boxes_ + 1) * sizeof (int),
														 cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->S_d_.Bn_, f->S_h_.Bn_,
														 f->S_h_.n_boxes_ * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__TARGET_BOX__ (FMMWrapper_t* f)
{
	/* Target boxes */
	cutilSafeCall (cudaMemcpy (f->T_d_.x_, f->T_h_.x_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->T_d_.y_, f->T_h_.y_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->T_d_.z_, f->T_h_.z_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->T_d_.w_, f->T_h_.w_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->T_d_.Bptr_, f->T_h_.Bptr_,
														 (f->T_h_.n_boxes_ + 1) * sizeof (int),
														 cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->T_d_.Bn_, f->T_h_.Bn_,
														 f->T_h_.n_boxes_ * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__U_LIST__ (FMMWrapper_t* f)
{
	/* Ulist */
	cutilSafeCall (cudaMemcpy (f->U_d_.L_, f->U_h_.L_,
														 f->U_h_.Ptr_[f->U_h_.n_boxes_] * 
														 sizeof (int), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->U_d_.Ptr_, f->U_h_.Ptr_,
														 (f->U_h_.n_boxes_ + 1) * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__TAG__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->tag_d_, f->tag_h_,
														 nodeVec.size () * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__DEPTH__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	/* depth */
	cutilSafeCall (cudaMemcpy (f->depth_d_, f->depth_h_,
														 nodeVec.size () * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__CHILDREN__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->child_d_, f->child_h_,
														 nodeVec.size () * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__RADIUS__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->radius_d_, f->radius_h_,
														 nodeVec.size () * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__CENTER__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->center0_d_, f->center0_h_,
														 nodeVec.size () * sizeof (dtype),
														 cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->center1_d_, f->center1_h_,
														 nodeVec.size () * sizeof (dtype),
														 cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->center2_d_, f->center2_h_,
														 nodeVec.size () * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__SP_UC__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->SP_UC_d_, f->SP_UC_h_,
														 3 * f->SP_UC_size_padded * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__UC2UE__ (FMMWrapper_t* f)
{
	/* UC2UE matrix */
	cutilSafeCall (cudaMemcpy (f->UC2UE_d_, f->UC2UE_h_,
														 f->UC2UE_r_padded * f->UC2UE_c * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__UE2UC__ (FMMWrapper_t* f)
{
	/* UE2UC matrix */
	cutilSafeCall (cudaMemcpy (f->UE2UC_d_, f->UE2UC_h_,
														 (2 * 2 * 2) * (f->UE2UC_r_padded * f->UE2UC_c) * 
														 sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__SRC_UPW_EQU_DEN__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->SRC_UPW_EQU_DEN_d_, f->SRC_UPW_EQU_DEN_h_,
														 nodeVec.size () * f->UC2UE_r_padded *
														 sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__VLIST_TLIST__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->vlist_d_, f->vlist_h_,
														 f->list_size * sizeof (int),
														 cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->tlist_d_, f->tlist_h_,
														 f->list_size * sizeof (int),
														 cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->vlist_ptr_d_, f->vlist_ptr_h_,
														 (nodeVec.size () + 1) * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__TRG_DWN_CHK_VAL__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->TRG_DWN_CHK_VAL_d_, f->TRG_DWN_CHK_VAL_h_,
														 nodeVec.size () * f->SP_DC_n_padded_ *
														 sizeof (dtype),
														 cudaMemcpyHostToDevice));
}


void
xfer__PATH2NODE__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->path2Node_d_, f->path2Node_h_,
														 nodeVec.size () * sizeof (int3),
														 cudaMemcpyHostToDevice));
}

void
xfer__PARENT__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->parent_d_, f->parent_h_,
														 nodeVec.size () * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__DC2DE__(FMMWrapper_t* f)
{
	/* DC2DE_mat */
	cutilSafeCall (cudaMemcpy (f->DC2DE_d_, f->DC2DE_h_,
														 f->DC2DE_r_padded * f->DC2DE_c * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__DE2DC__(FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->DE2DC_d_, f->DE2DC_h_,
														 (2 * 2 * 2) * f->DE2DC_r_padded * f->DE2DC_c *
														 sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__SP_DE__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->SP_DE_d_, f->SP_DE_h_,
														 3 * f->SP_DE_n_padded * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__SP_UE__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->SP_UE_d_, f->SP_UE_h_,
														 3 * f->SP_UE_n_padded * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__W_LIST__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->W_d_.L_, f->W_h_.L_,
														 f->W_h_.Ptr_[f->W_h_.n_boxes_] * 
														 sizeof (int), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->W_d_.Ptr_, f->W_h_.Ptr_,
														 (f->W_h_.n_boxes_ + 1) * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__SRCNUM__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->srcNum_d_, f->srcNum_h_,
														 nodeVec.size () * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__SP_DC__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->SP_DC_d_, f->SP_DC_h_,
														 3 * f->SP_DC_n_padded_ * sizeof (dtype),
														 cudaMemcpyHostToDevice));
}

void
xfer__X_LIST__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->X_d_.L_, f->X_h_.L_,
														 f->X_h_.Ptr_[f->X_h_.n_boxes_] * 
														 sizeof (int), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (f->X_d_.Ptr_, f->X_h_.Ptr_,
														 (f->X_h_.n_boxes_ + 1) * sizeof (int),
														 cudaMemcpyHostToDevice));
}

void
xfer__TRGNUM__ (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->trgNum_d_, f->trgNum_h_,
														 nodeVec.size () * sizeof (int),
														 cudaMemcpyHostToDevice));
}


void
xfer__SRC_UPW_EQU_DEN__back (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->SRC_UPW_EQU_DEN_h_, f->SRC_UPW_EQU_DEN_d_,
														 nodeVec.size () * f->UC2UE_r_padded *
														 sizeof (dtype),
														 cudaMemcpyDeviceToHost));
}

void
xfer__TRG_DWN_CHK_VAL__back (FMMWrapper_t* f)
{
  AllNodes *All_N = f->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

	cutilSafeCall (cudaMemcpy (f->TRG_DWN_CHK_VAL_h_, f->TRG_DWN_CHK_VAL_d_,
														 nodeVec.size () * f->SP_DC_n_padded_ *
														 sizeof (dtype),
														 cudaMemcpyDeviceToHost));
}


void
xfer__TARGET_BOX__back (FMMWrapper_t* f)
{
	cutilSafeCall (cudaMemcpy (f->T_h_.w_, f->T_d_.w_,
														 f->T_h_.Bptr_[f->T_h_.n_boxes_] * 
														 sizeof (dtype), cudaMemcpyDeviceToHost));
}


void
free__SOURCE_BOX__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* Deallocate memory for data */
	cutilSafeCall (cudaFree (f->S_d_.x_));
	cutilSafeCall (cudaFree (f->S_d_.y_));
	cutilSafeCall (cudaFree (f->S_d_.z_));
	cutilSafeCall (cudaFree (f->S_d_.w_));
	/* Deallocate memory for pointers */
	cutilSafeCall (cudaFree (f->S_d_.Bptr_));
	cutilSafeCall (cudaFree (f->S_d_.Bn_));
  /* ------------------------------------------------------------ */
}

void
free__TARGET_BOX__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* Target boxes */
	cutilSafeCall (cudaFree (f->T_d_.x_));
	cutilSafeCall (cudaFree (f->T_d_.y_));
	cutilSafeCall (cudaFree (f->T_d_.z_));
	cutilSafeCall (cudaFree (f->T_d_.w_));
	/* Allocate memory for pointers */
	cutilSafeCall (cudaFree (f->T_d_.Bptr_));
	cutilSafeCall (cudaFree (f->T_d_.Bn_));
  /* ------------------------------------------------------------ */
}

void
free__U_LIST__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* ulist */
	cutilSafeCall (cudaFree (f->U_d_.L_));
	cutilSafeCall (cudaFree (f->U_d_.Ptr_));
  /* ------------------------------------------------------------ */
}

void
free__TAG__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* tag */
	cutilSafeCall (cudaFree (f->tag_d_));
  /* ------------------------------------------------------------ */
}


void
free__DEPTH__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* depth */
	cutilSafeCall (cudaFree (f->depth_d_));
  /* ------------------------------------------------------------ */
}


void
free__CHILDREN__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* children */
	cutilSafeCall (cudaFree (f->child_d_));
  /* ------------------------------------------------------------ */
}


void
free__RADIUS__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* radius */
	cutilSafeCall (cudaFree (f->radius_d_));
  /* ------------------------------------------------------------ */
}

void
free__CENTER__ (FMMWrapper_t* f) {
	/* ------------------------------------------------------------ */
	cutilSafeCall (cudaFree (f->center0_d_));
	cutilSafeCall (cudaFree (f->center1_d_));
	cutilSafeCall (cudaFree (f->center2_d_));
  /* ------------------------------------------------------------ */
}

void
free__SP_UC__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* SP[UC] */
	cutilSafeCall (cudaFree (f->SP_UC_d_));
  /* ------------------------------------------------------------ */
}


void
free__UC2UE__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* UC2UE matrix */
	cutilSafeCall (cudaFree (f->UC2UE_d_));
  /* ------------------------------------------------------------ */
}

void
free__UE2UC__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* UE2UC matrix */
	cutilSafeCall (cudaFree (f->UE2UC_d_));
  /* ------------------------------------------------------------ */
}


void
free__SRC_UPW_EQU_DEN__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* Temporary up_calc GPU variables */
	/* src_upw_equ_den */
	cutilSafeCall (cudaFree (f->SRC_UPW_EQU_DEN_d_));
  /* ------------------------------------------------------------ */

}


void
free__VLIST_SRC__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* src */
	cutilSafeCall (cudaFree (f->vlist_src_d_));
  /* ------------------------------------------------------------ */
}

void
free__REG_DEN__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
  /* reg_den */
	cutilSafeCall (cudaFree (f->reg_den_d_));
  /* ------------------------------------------------------------ */
}

void
free__TT__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* tt */
	cutilSafeCall (cudaFree (f->tt));
  /* ------------------------------------------------------------ */
}

void
free__VLIST_TRANS__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
  /* trans */
	cutilSafeCall (cudaFree (f->vlist_trans_d_));
  /* ------------------------------------------------------------ */
}


void
free__VLIST_TRG__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* trg */
	cutilSafeCall (cudaFree (f->vlist_trg_d_));
  /* ------------------------------------------------------------ */
}

void
free__VLIST_TLIST__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* vlist and tlist and pointer */
	/* pointer */
	cutilSafeCall (cudaFree (f->vlist_ptr_d_));
	/* vlist */
	cutilSafeCall (cudaFree (f->vlist_d_));
	/* tlist */
	cutilSafeCall (cudaFree (f->tlist_d_));
  /* ------------------------------------------------------------ */
}

void
free__TRG_DWN_CHK_VAL__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
  /* trg_dwn_chk_val */
	cutilSafeCall (cudaFree (f->TRG_DWN_CHK_VAL_d_));
  /* ------------------------------------------------------------ */
}

void
free__PATH2NODE__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* path2Node */
	cutilSafeCall (cudaFree (f->path2Node_d_));
  /* ------------------------------------------------------------ */
}

void
free__PARENT__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* parent */
	/* Not needed - use children structure instead */
	/* Now I need it */
	cutilSafeCall (cudaFree (f->parent_d_));
  /* ------------------------------------------------------------ */
}

void
free__TRG_DWN_EQU_DEN__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* trg_dwn_equ_den */
	cutilSafeCall (cudaFree (f->TRG_DWN_EQU_DEN_d_));
  /* ------------------------------------------------------------ */
}

void
free__DC2DE__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* DC2DE_mat */
	cutilSafeCall (cudaFree (f->DC2DE_d_));
  /* ------------------------------------------------------------ */
}

void
free__DE2DC__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* DE2DC_mat[8] */
	cutilSafeCall (cudaFree (f->DE2DC_d_));
  /* ------------------------------------------------------------ */
}

void
free__SP_DE__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* SP[DE] */
	cutilSafeCall (cudaFree (f->SP_DE_d_));
  /* ------------------------------------------------------------ */
}

void
free__SP_UE__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaFree (f->SP_UE_d_));
}

void
free__W_LIST__ (FMMWrapper_t* f) 
{
  /* ------------------------------------------------------------ */
	/* ulist */
	cutilSafeCall (cudaFree (f->W_d_.L_));
	cutilSafeCall (cudaFree (f->W_d_.Ptr_));
  /* ------------------------------------------------------------ */
}

void
free__SRCNUM__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaFree (f->srcNum_d_));
}

void
free__SP_DC__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaFree (f->SP_DC_d_));
}

void
free__X_LIST__ (FMMWrapper_t* f)
{
  /* ------------------------------------------------------------ */
	/* ulist */
	cutilSafeCall (cudaFree (f->X_d_.L_));
	cutilSafeCall (cudaFree (f->X_d_.Ptr_));
  /* ------------------------------------------------------------ */
}

void
free__TRGNUM__ (FMMWrapper_t* f)
{
	cutilSafeCall (cudaFree (f->trgNum_d_));
}







FMMWrapper_t *
preproc (AllNodes* All_N)
{
  FMMWrapper_t* f = (FMMWrapper_t *) malloc (sizeof (FMMWrapper_t));
  assert (f);

  f->AN = All_N;

	int i, j, idx;

	Point3 c;
	real_t r;
	// int num_leaf_nodes;
	// int num_non_leaf_nodes;

	int num_leaf_nodes_src;
	int num_leaf_nodes_trg;
	int num_non_leaf_nodes_src;
	int num_non_leaf_nodes_trg;

	int list_size;

	Pos *SP = All_N->SP;
	Trans_matrix *TM = All_N->TM;

	Pos *RP = All_N->RP;

  struct stopwatch_t* timer = NULL;
  struct stopwatch_t* timer_ = NULL;
  long double t_data_cpu, t_data_gpu, t_pcie, t_subtract;
  stopwatch_init ();
  timer = stopwatch_create ();
  timer_ = stopwatch_create ();


	/* ============================================================= */
	/* CPU SIDE 
	 */
	real_t* tmp_c;
	tmp_c = (real_t*) malloc (1024 * 1024);
	assert (tmp_c);

  fprintf (stderr, "Creating Host Data Structures ... ");
  stopwatch_start (timer);

	/* byte alignment required for coalesced loading */
	int byte_padding = get_byte_padding ();
	/* real_padding is padding in terms of # of data elements */
	int real_padding = byte_padding / sizeof (dtype);

	/* Create GPU friendly Source boxes */
	vector<NodeTree>& nodeVec = *All_N->N;
	/*
	ulist_create_boxes__double (&f->S_h_, nodeVec.size (), 
															All_N->Ns, real_padding);	
	 */
	ulist_create_boxes__double_source (All_N, f);

	/* Create GPU friendly Target boxes */
	/*
	ulist_create_boxes__double (&f->T_h_, nodeVec.size (), 
															All_N->Nt, real_padding);	
	 */
	ulist_create_boxes__double_target (All_N, f);

	/* Create GPU friendly ulist data structure */
	ulist_create_ulist (&f->U_h_, nodeVec.size (), All_N);	



	
	/* depth and children */
	f->depth_h_ = (int*) malloc (nodeVec.size () * sizeof (int));
	assert (f->depth_h_);
	for(i = 0; i < nodeVec.size (); i++) {
		f->depth_h_[i] = nodeVec[i].depth;
	}

	// num_leaf_nodes = 0;
	// num_non_leaf_nodes = 0;
	/*
	num_leaf_nodes = (int) pow (8.0, nodeVec[nodeVec.size () - 1].depth);	
	num_non_leaf_nodes = nodeVec.size () - num_leaf_nodes;
	assert ((num_leaf_nodes + num_non_leaf_nodes) == nodeVec.size ());
	 */
	/* num leaf and non-leaf nodes for src and trg */
	num_leaf_nodes_src = 0;
	num_non_leaf_nodes_src = 0;
	for(i = 0; i < nodeVec.size (); i++) {
		if(nodeVec[i].tag & LET_SRCNODE && nodeVec[i].child == -1) {
			num_leaf_nodes_src++; 
		} else if(nodeVec[i].tag & LET_SRCNODE && nodeVec[i].child != -1) {
			num_non_leaf_nodes_src++;
		} 
	}
	num_leaf_nodes_trg = 0;
	num_non_leaf_nodes_trg = 0;
	for(i = 0; i < nodeVec.size (); i++) {
		if(nodeVec[i].tag & LET_TRGNODE && nodeVec[i].child == -1) {
			num_leaf_nodes_trg++; 
		} else if(nodeVec[i].tag & LET_SRCNODE && nodeVec[i].child != -1) {
			num_non_leaf_nodes_trg++;
		} 
	}

	f->child_h_ = (int*) malloc (nodeVec.size () * sizeof (int));
	assert (f->child_h_);
	for(i = 0; i < nodeVec.size (); i++) {
		f->child_h_[i] = nodeVec[i].child;
	}


	/* Pre-compute center and radius */
	f->radius_h_ = (real_t*) malloc (nodeVec.size () * sizeof (real_t));
	f->center0_h_ = (real_t*) malloc (nodeVec.size () * sizeof (real_t));
	f->center1_h_ = (real_t*) malloc (nodeVec.size () * sizeof (real_t));
	f->center2_h_ = (real_t*) malloc (nodeVec.size () * sizeof (real_t));
	assert (f->radius_h_ && f->center0_h_ && f->center1_h_ && f->center2_h_);
	// idx = 0;
	for(i = 0; i < nodeVec.size (); i++) {

			c = center (i, nodeVec);
			r = radius (i, nodeVec);

			f->radius_h_[i] = r;
			f->center0_h_[i] = c(0);
			f->center1_h_[i] = c(1);
			f->center2_h_[i] = c(2);
	}


	/* tag */
	/* SRC or TG */
	f->tag_h_ = (int*) malloc (nodeVec.size () * sizeof (int));
	for(i = 0; i < nodeVec.size (); i++) {
		f->tag_h_[i] = nodeVec[i].tag;
	}

	/* SP[UC] */
	/* SP[UC] consists of 3 arrays x, y, and z each of which are
	 * (np+2)^3 - (np)^3 long
	 */
	/* allocate memory */
	f->SP_UC_size = pln_size (UC, SP);
	f->SP_UC_size_padded = (((pln_size (UC, SP) + real_padding - 1) / 
													real_padding) * real_padding);
	/* 3x for x, y, and z */
	f->SP_UC_h_ = (real_t*) malloc (3 * f->SP_UC_size_padded * sizeof (real_t));
	assert (f->SP_UC_h_);
	/* initialize data */
	memcpy (&f->SP_UC_h_[0], SP[UC].x, SP[UC].n * sizeof (real_t));
	memcpy (&f->SP_UC_h_[1 * f->SP_UC_size_padded], SP[UC].y, 
					SP[UC].n * sizeof (real_t));
	memcpy (&f->SP_UC_h_[2 * f->SP_UC_size_padded], SP[UC].z, 
					SP[UC].n * sizeof (real_t));

	/* UC2UE */
	stopwatch_start (timer_);
	compute_UC2UE_mat (TM, SP);	
	t_subtract = stopwatch_stop (timer_);

	f->UC2UE_r = pln_size (UE, SP);	
	f->UC2UE_r_padded = (((f->UC2UE_r + real_padding - 1) / real_padding) * 
											 real_padding);
	f->UC2UE_c = pln_size (UC, SP);	
	f->UC2UE_h_ = (real_t*) malloc (f->UC2UE_r_padded * f->UC2UE_c * 
																	sizeof (real_t));
	assert (f->UC2UE_h_);
	for(i = 0; i < f->UC2UE_c; i++) {
		memcpy (&f->UC2UE_h_[i * f->UC2UE_r_padded], &TM->UC2UE[i * f->UC2UE_r],
						f->UC2UE_r * sizeof (real_t));
	}


	/* UE2UC */
	stopwatch_start (timer_);
	TM->UE2UC = (real_t**) malloc (sizeof (real_t*) * 2 * 2 * 2);
	for(int a_ = 0; a_ < 2; a_++) {
		for(int b_ = 0; b_ < 2; b_++) {
			for(int c_ = 0; c_ < 2; c_++) {
				Index3 idx3(a_, b_, c_);
				compute_UE2UC_mat (idx3, TM, SP);
			}
		}	
	}
	t_subtract += stopwatch_stop (timer_);

	f->UE2UC_r = pln_size (UC, SP);
	f->UE2UC_r_padded = (((f->UE2UC_r + real_padding - 1) / real_padding) *
											 real_padding);
	f->UE2UC_c = pln_size (UE, SP);
	f->UE2UC_h_ = (real_t*) malloc ((2 * 2 * 2) * (f->UE2UC_r_padded * f->UE2UC_c)
																	*  sizeof (real_t));
	assert (f->UE2UC_h_);
	/* copy each matrix */
	for(i = 0; i < (2 * 2 * 2); i++) {
		/* 1 column at a time */
		for(j = 0; j < f->UE2UC_c; j++) {
			memcpy (&f->UE2UC_h_[i * f->UE2UC_r_padded * f->UE2UC_c + 
													 j * f->UE2UC_r_padded], 
							&TM->UE2UC[i][j * f->UE2UC_r], f->UE2UC_r * sizeof (dtype));
		}
	}

	/* SRC_UPW_EQU_DEN_h_ */
	f->SRC_UPW_EQU_DEN_h_ = (dtype*) malloc (nodeVec.size () * f->UC2UE_r_padded *
																					 sizeof (dtype));
	assert (f->SRC_UPW_EQU_DEN_h_);

	
	/* up_calc__gpu configuration variables */
	// f->num_non_leaf_nodes = num_non_leaf_nodes;
	f->tree_max_depth = nodeVec[nodeVec.size () - 1].depth;
	f->reduction_depth = f->tree_max_depth - 2;
	f->num_nodes_reduction = (int) pow (8.0, f->reduction_depth);
	f->reduction_offset = 0;
	for(i = 0; i < f->reduction_depth; i++) 
		f->reduction_offset += (int) pow (8.0, i);

	/* up_calc temporary arrays */
	/* src_upw_equ_den */
	/* There is no CPU equivalent of this as all this is needed is in the GPU */


	/* VLIST data structures */
	f->vlist_array_size = eff_data_size (UE);
	/*
	f->vlist_array_size_padded = (((f->vlist_array_size + real_padding - 1) / 
															 real_padding) * real_padding);
	 */
	/* trg */
	/* src */
	f->vlist_ptr_h_ = (int*) malloc ((nodeVec.size () + 1) * sizeof (int));
	assert (f->vlist_ptr_h_);

	list_size = 0;
	f->vlist_ptr_h_[0] = 0;
	for(i = 0; i < nodeVec.size (); i++) {
			list_size += nodeVec[i].Vnodes.size ();
			f->vlist_ptr_h_[i+1] = list_size;
	}
	f->vlist_h_ = (int*) malloc (list_size * sizeof (int));
	assert (f->vlist_h_);

	idx = 0;
	for(i = 0; i < nodeVec.size (); i++) {
		for(j = 0; j < nodeVec[i].Vnodes.size (); j++) {
			f->vlist_h_[idx] = nodeVec[i].Vnodes[j];
			idx++;
		}	
	}
	assert (idx == list_size);
	f->list_size = list_size;
	
	f->reg_den_size = RP->n;
	/*
	f->reg_den_size_padded = (((RP->n + real_padding - 1) / real_padding) * 
													 real_padding);
	 */
	/* reg den needs no host equivalent */
	
	/* trans */
	f->trans_arrays_num = 7 * 7 * 7;

	f->tlist_h_ = (int*) malloc (list_size * sizeof (int));
	assert (f->tlist_h_);
		
	int id;	
	int dim = 3;
	int t_index = 0;
	for(i = 0;i < nodeVec.size (); i++) {
		if(nodeVec[i].tag & LET_TRGNODE && nodeVec[i].Vnodes.size () > 0) {
			Point3 gNodeIdxCtr (center (i, nodeVec));
			real_t D = 2.0 * radius (i, nodeVec);
			for(j = 0;j < nodeVec[i].Vnodes.size (); j++) {
				idx = nodeVec[i].Vnodes[j];
				Point3 viCtr (center (idx, nodeVec));
				Index3 idx3;
				for(int d = 0; d < dim; d++) {
					idx3(d) = int (round ((viCtr[d] - gNodeIdxCtr[d]) / D));
				}
				id = (idx3(0) + 3) + (idx3(1) + 3) * 7 + (idx3(2) + 3) * 7 * 7;
				f->tlist_h_[t_index] = id;
				t_index++;
			}
		}
	}


	/* No need for these on the host */
	/* f->vlist_src_d_ */
	/* f->vlist_trg_d_ */
	/* f->vlist_trans_d_ */


	f->RP_n_ = RP->n;
	/* No need for these on the host */
	/* f->RP_X_d_ */
	/* f->RP_Y_d_ */
	/* f->RP_Z_d_ */


	/* IFFT */
	f->SP_DC_n_ = pln_size (DC, SP);
	f->SP_DC_n_padded_ = (((f->SP_DC_n_ + real_padding - 1) / real_padding) *
												real_padding);

	/* TRG_DWN_CHK_VAL_h_ */
	f->TRG_DWN_CHK_VAL_h_ = (dtype*) malloc (nodeVec.size () * 
																					 f->SP_DC_n_padded_ * sizeof (dtype));
	assert (f->TRG_DWN_CHK_VAL_h_);



	/* DOWN_CALC */
	/* path2Node */
	f->path2Node_h_ = (int3*) malloc (nodeVec.size () * sizeof (int3));
	assert (f->path2Node_h_);
	for(i = 0; i < nodeVec.size (); i++) {
		f->path2Node_h_[i].x = (nodeVec[i].path2Node)(0);
		f->path2Node_h_[i].y = (nodeVec[i].path2Node)(1);
		f->path2Node_h_[i].z = (nodeVec[i].path2Node)(2);
	}
	/* parent */	
	/* Not needed - use children structure instead */
	/* Actually, now I need it */
	f->parent_h_ = (int*) malloc (nodeVec.size () * sizeof (int));
	assert (f->parent_h_);
	for(i = 0; i < nodeVec.size (); i++) {
		f->parent_h_[i] = nodeVec[i].parent;
	}

	/* trg_dwn_equ_den */
	f->SP_DE_n_ = pln_size (DE, SP);	
	f->SP_DE_n_padded = (((f->SP_DE_n_ + real_padding - 1) / real_padding) *
											 real_padding);
	/* DC2DE_mat */
	stopwatch_start (timer_);
	compute_DC2DE_mat (TM, SP);
	t_subtract += stopwatch_stop (timer_);

	f->DC2DE_r = pln_size (DE, SP);
	f->DC2DE_r_padded = (((f->DC2DE_r + real_padding - 1) / real_padding) *
											 real_padding);
	f->DC2DE_c = pln_size (DC, SP);
	f->DC2DE_h_ = (real_t*) malloc (f->DC2DE_r_padded * f->DC2DE_c * 
																	sizeof (real_t));
	assert (f->DC2DE_h_);
	for(i = 0; i < f->DC2DE_c ; i++) {
		memcpy (&f->DC2DE_h_[i * f->DC2DE_r_padded],
						&TM->DC2DE[i * f->DC2DE_r], 
						f->DC2DE_r * sizeof (real_t));
	}
	/* DE2DC_mat[8] */
	stopwatch_start (timer_);
	TM->DE2DC = (real_t**) malloc (sizeof (real_t*) * 2 * 2 * 2);
	for(int a = 0; a < 2; a++) {
		for(int b = 0; b < 2; b++) {
			for(int c = 0; c < 2 ; c++) {
				Index3 idx(a, b, c);
				compute_DE2DC_mat (idx, TM, SP);
			}
		}
	}
	t_subtract += stopwatch_stop (timer_);

	f->DE2DC_r = pln_size (DC, SP);
	f->DE2DC_r_padded = (((f->DE2DC_r + real_padding - 1) / real_padding) *
											 real_padding);
	f->DE2DC_c = pln_size (DE, SP);
	f->DE2DC_h_ = (real_t*) malloc ((2 * 2 * 2) * f->DE2DC_r_padded * f->DE2DC_c *
																	sizeof (real_t));
	assert (f->DE2DC_h_);
	for(i = 0; i < 2 * 2 * 2; i++) {
		for(j = 0; j < f->DE2DC_c; j++) {
			real_t* temp_tm = TM->DE2DC[i];
			memcpy (&f->DE2DC_h_[i * f->DE2DC_r_padded * f->DE2DC_c + 
													 j * f->DE2DC_r_padded],
							&temp_tm[j * f->DE2DC_r],
							f->DE2DC_r * sizeof (real_t));
		}
	}
	/* down_calc configuration */
	f->expansion_depth = 2;
	f->num_nodes_expansion = (int) pow (8.0, f->expansion_depth);
	f->expansion_offset = 0;
	for(i = 0; i < f->expansion_depth; i++) {
		f->expansion_offset += (int) pow (8.0, i);
	}

	/* down_calc SP[DE] */
	f->SP_DE_h_ = (dtype*) malloc (3 * f->SP_DE_n_padded * sizeof (dtype));
	assert (f->SP_DE_h_);
	memcpy (&f->SP_DE_h_[0], SP[DE].x, SP[DE].n * sizeof (dtype));
	memcpy (&f->SP_DE_h_[f->SP_DE_n_padded], SP[DE].y, SP[DE].n * sizeof (dtype));
	memcpy (&f->SP_DE_h_[2 * f->SP_DE_n_padded], SP[DE].z, 
					SP[DE].n * sizeof (dtype));

  //t_data_cpu = stopwatch_stop (timer) - t_subtract;
  t_data_cpu = stopwatch_stop (timer); 
  fprintf (stderr, "==> Time: %Lg secs\n", t_data_cpu);


	/* WLIST_CALC */
	f->SP_UE_n_ = pln_size (UE, SP);	
	f->SP_UE_n_padded = (((f->SP_UE_n_ + real_padding - 1) / real_padding) *
											 real_padding);

	f->SP_UE_h_ = (dtype*) malloc (3 * f->SP_UE_n_padded * sizeof (dtype));
	assert (f->SP_UE_h_);
	memcpy (&f->SP_UE_h_[0], SP[UE].x, SP[UE].n * sizeof (dtype));
	memcpy (&f->SP_UE_h_[f->SP_UE_n_padded], SP[UE].y, SP[UE].n * sizeof (dtype));
	memcpy (&f->SP_UE_h_[2 * f->SP_UE_n_padded], SP[UE].z, 
					SP[UE].n * sizeof (dtype));

	wlist_create_wlist (&f->W_h_, nodeVec.size (), All_N);	


	f->srcNum_h_ = (int*) malloc (nodeVec.size () * sizeof (int));
	assert (f->srcNum_h_);
	for(i = 0; i < nodeVec.size (); i++) {
		f->srcNum_h_[i] = nodeVec[i].srcNum;
	}


	/* XLIST_CALC */
	f->SP_DC_h_ = (dtype*) malloc (3 * f->SP_DC_n_padded_ * sizeof (dtype));	
	assert (f->SP_DC_h_);

	memcpy (&f->SP_DC_h_[0], SP[DC].x, SP[DC].n * sizeof (dtype));
	memcpy (&f->SP_DC_h_[f->SP_DC_n_padded_], SP[DC].y, SP[DC].n * sizeof (dtype));
	memcpy (&f->SP_DC_h_[2 * f->SP_DC_n_padded_], SP[DC].z, 
					SP[DC].n * sizeof (dtype));

	xlist_create_xlist (&f->X_h_, nodeVec.size (), All_N);

	f->trgNum_h_ = (int*) malloc (nodeVec.size () * sizeof (int));
	assert (f->trgNum_h_);
	for(i = 0; i < nodeVec.size (); i++) {
		f->trgNum_h_[i] = nodeVec[i].trgNum;
	}




	#if 0
		long int bytes_up = 0;
		/* source boxes */
		bytes_up += 4 * f->S_h_.Bptr_[f->S_h_.n_boxes_] * sizeof (dtype);
		bytes_up += (f->S_h_.n_boxes_ + 1) * sizeof (int);
		bytes_up += (f->S_h_.n_boxes_) * sizeof (int);
	
		/* Radius */
		bytes_up += nodeVec.size () * sizeof (dtype);

		/* center */
		bytes_up += 3 * nodeVec.size () * sizeof (dtype);

		/* SP_UC */
		bytes_up += 3 * f->SP_UC_size_padded * sizeof (dtype);

		/* UC2UE */
		bytes_up += f->UC2UE_r_padded * f->UC2UE_c * sizeof (dtype);

		/* src_upw_equ_den */
		bytes_up += nodeVec.size () * f->UC2UE_r_padded * sizeof (dtype);

		/* child */
		bytes_up += nodeVec.size () * sizeof (int);

		/* UE2UC */
		bytes_up += 8 * f->UE2UC_r_padded * f->UE2UC_c * sizeof (dtype);

		/* tag */
		bytes_up += nodeVec.size () * sizeof (int);

		/* depth */
		bytes_up += nodeVec.size () * sizeof (int);

    double mega_bytes_up = (1.0 * bytes_up/ 1000000);
    printf("VLIST requires %g mega bytes of data\n", mega_bytes_up);
	#endif

	#if 0
    long int bytes_vlist = 0;
    /* DEPTH */
    bytes_vlist += nodeVec.size () * sizeof (int);
    /* SRC_UPW_EQU_DEN */
    bytes_vlist += nodeVec.size () * f->UC2UE_r_padded * sizeof (dtype);
    /* REG_DEN */
    bytes_vlist += nodeVec.size () * f->reg_den_size * sizeof (dtype);
    /* VLIST_SRC */
    bytes_vlist += nodeVec.size () * f->vlist_array_size * sizeof (dtype);
    /* TT */
    bytes_vlist += f->trans_arrays_num * f->RP_n_ * sizeof (dtype);
    /* VLIST_TRANS */
    bytes_vlist += f->trans_arrays_num * f->vlist_array_size * sizeof (dtype);
    /* VLIST_TRG */
    bytes_vlist += nodeVec.size () * f->vlist_array_size * sizeof (dtype);

    /* VLIST_TLIST */
    bytes_vlist += (nodeVec.size () + 1) * sizeof (int);
    bytes_vlist += list_size * sizeof (int);
    bytes_vlist += list_size * sizeof (int);

    /* REG_DEN_IFFT */
    bytes_vlist += nodeVec.size () * f->reg_den_size * sizeof (dtype);
    /* TRG_DWN_CHK_VAL */
    bytes_vlist += nodeVec.size () * f->SP_DC_n_padded_ * sizeof (dtype);

    double mega_bytes_vlist = (1.0 * bytes_vlist / 1000000);
    printf("VLIST requires %g mega bytes of data\n", mega_bytes_vlist);
	#endif


	/* ============================================================= */
	/* GPU SIDE 
	 */
	/* this is done to set up the GPU */
	real_t* tmp_g;
	cutilSafeCall (cudaMalloc ((void**)&tmp_g, 1024 * 1024));
	
  fprintf (stderr, "Creating GPU Data Structures ... ");
  stopwatch_start (timer);



  #if __SOURCE_BOX__
	alloc__SOURCE_BOX__ (f);
  #endif

  #if __TARGET_BOX__
	alloc__TARGET_BOX__ (f);
  #endif

  #if __U_LIST__
	alloc__U_LIST__ (f);
  #endif

	#if __TAG__
	alloc__TAG__ (f);
	#endif

  #if __DEPTH__
	alloc__DEPTH__ (f);
  #endif

  #if __CHILDREN__
	alloc__CHILDREN__ (f);
  #endif
  
  #if __RADIUS__
	alloc__RADIUS__ (f);
  #endif

  #if __CENTER__
	alloc__CENTER__ (f);
  #endif

  #if __SP_UC__
	alloc__SP_UC__ (f);
  #endif

  #if __UC2UE__
	alloc__UC2UE__ (f);
  #endif

  #if __UE2UC__
	alloc__UE2UC__ (f);
  #endif

  #if __SRC_UPW_EQU_DEN__
	alloc__SRC_UPW_EQU_DEN__ (f);
  #endif

	/* Vlist */
  #if __VLIST_SRC__
	alloc__VLIST_SRC__ (f);
  #endif

  #if __REG_DEN__
	alloc__REG_DEN__ (f);
  #endif
  
  #if __TT__
	alloc__TT__ (f);
  #endif

  #if __VLIST_TRANS__
	alloc__VLIST_TRANS__ (f);
  #endif

  #if __VLIST_TRG__
	alloc__VLIST_TRG__ (f);
  #endif

  #if __VLIST_TLIST__
	alloc__VLIST_TLIST__ (f);
  #endif

	#if 0
  #if __REG_DEN_IFFT__
  /* ------------------------------------------------------------ */
	/* IFFT */	
	cutilSafeCall (cudaMalloc ((void**)&f->reg_den_ifft_d_,
														 nodeVec.size () * f->reg_den_size *
														 sizeof (dtype)));
	assert (f->reg_den_ifft_d_);
  /* ------------------------------------------------------------ */
  #endif
	#endif
	
  #if __TRG_DWN_CHK_VAL__	
	alloc__TRG_DWN_CHK_VAL__ (f);
  #endif

	/* DOWN_CALC */
  #if __PATH2NODE__
	alloc__PATH2NODE__ (f);
  #endif

	#if __PARENT__
	alloc__PARENT__ (f);
	#endif

  #if __TRG_DWN_EQU_DEN__
	alloc__TRG_DWN_EQU_DEN__ (f);
  #endif

  #if __DC2DE__
	alloc__DC2DE__ (f);
  #endif

  #if __DE2DC__
	alloc__DE2DC__ (f);
  #endif

  #if __SP_DE__
	alloc__SP_DE__ (f);
  #endif

	#if __SP_UE__
	alloc__SP_UE__ (f);
	#endif


  #if __W_LIST__
	alloc__W_LIST__ (f);
  #endif

	#if __SRCNUM__
	alloc__SRCNUM__ (f);
	#endif

	#if __SP_DC__
	alloc__SP_DC__ (f);
	#endif


  #if __X_LIST__
	alloc__X_LIST__ (f);
  #endif

	#if __TRGNUM__
	alloc__TRGNUM__ (f);
	#endif

  t_data_gpu = stopwatch_stop (timer);
  fprintf (stderr, "==> Time: %Lg secs\n", t_data_gpu);


	/* ============================================================= */
	/* Copy data over to GPU 
	 */

  fprintf (stderr, "Copying Data over PCIE ... ");
  stopwatch_start (timer);

  #if __SOURCE_BOX__
	xfer__SOURCE_BOX__ (f);
  #endif

  #if __TARGET_BOX__
	xfer__TARGET_BOX__ (f);
  #endif

  #if __U_LIST__
	xfer__U_LIST__ (f);
  #endif

	#if __TAG__
	xfer__TAG__ (f);
	#endif

  #if __DEPTH__
	xfer__DEPTH__ (f);
  #endif

  #if __CHILDREN__
	xfer__CHILDREN__ (f);
  #endif

	/* center and radius */
  #if __RADIUS__
	xfer__RADIUS__ (f);
  #endif

  #if __CENTER__
	xfer__CENTER__ (f);
  #endif

	/* SP[UC] */
  #if __SP_UC__
	xfer__SP_UC__ (f);
  #endif

  #if __UC2UE__
	xfer__UC2UE__ (f);
  #endif
  
  #if __UE2UC__
	xfer__UE2UC__ (f);
  #endif

	/* No copying necessary for SRC_UPW_EQU_DEN_d_ */
	/* No copying necessary for vlist_src_d_, vlist_trg_d_, vlist_trans_d_*/
	/* No copying necessary for tt and reg_den */

	/* vlist, tlist and pointer */
  #if __VLIST_TLIST__
	xfer__VLIST_TLIST__ (f);
  #endif

	/* No copying necessary for reg_den_ifft_d_ */
	/* No copying necessary for TRG_DWN_CHK_VAL_d_ */

	/* DOWN_CALC */
	/* path2Node */
  #if __PATH2NODE__
	xfer__PATH2NODE__ (f);
  #endif

	/* No copying necessary for TRG_DWN_EQU_DEN_d_ */

	/* parent */
	/* Not needed - use children structure instead */
	/* Now I need it */
	#if __PARENT__
	xfer__PARENT__ (f);
	#endif

  #if __DC2DE__
	xfer__DC2DE__ (f);
  #endif

	/* DE2DC_mat[8] */
  #if __DE2DC__
	xfer__DE2DC__ (f);
  #endif

	/* SP[DE] */
  #if __SP_DE__
	xfer__SP_DE__ (f);
  #endif

	/* SP[UE] */
	#if __SP_UE__
	xfer__SP_UE__ (f);
	#endif

  #if __W_LIST__
	xfer__W_LIST__ (f);
  #endif

	#if __SRCNUM__
	xfer__SRCNUM__ (f);
	#endif


	#if __SP_DC__
	xfer__SP_DC__ (f);
	#endif

  #if __X_LIST__
	xfer__X_LIST__ (f);
  #endif

	#if __TRGNUM__
	xfer__TRGNUM__ (f);
	#endif


  t_pcie = stopwatch_stop (timer);
  fprintf (stderr, "==> Time: %Lg secs\n", t_pcie);



  return f;
}

