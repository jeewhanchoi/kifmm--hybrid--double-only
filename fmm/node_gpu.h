#if !defined (INC_NODE_GPU_H)
#define INC_NODE_GPU_H

#include "reals.h"

typedef int int__gpu__t;

#if !defined (__CUDACC__)
typedef point4_t float4;
#endif

#define WARP_SIZE 32

#if defined (__cplusplus)
extern "C" {
#endif

#define gpu_check_error(fp)  gpu_check_error__srcpos (fp, __FILE__, __LINE__)

#define dtype double
  /**
   *  \name Points, grouped into boxes.
   *
   *  Position of point i == (x[i], y[i], z[i]).
   *  Density/Potential of point i == w[i].
   *
   *  Box k consists of the points (x[i..j], y[i..j], z[i..j]) where
   *  i == Bptr_[k] and j == i + Bn_[k] - 1 < Bptr_[k+1]
   *
   *  Note: n_points_ == No. of "true" (physical) points, whereas
   *  Bptr_[n_boxes_] == No. of stored points (length of x, y, z, and
   *  w arrays), including padding.
   */
  typedef struct
  {
    int n_points_;  /* total number of points for all nodes/boxes in tree */
    real_t* x_;
    real_t* y_;
    real_t* z_;
    real_t* w_;

    int n_boxes_;   /* total number of boxes/nodes in tree */
    int* Bptr_;     /* index into beginning of points for each node/box */
    int* Bn_;       /* number of points in each box */
  } Boxes_t;

  typedef struct
  {
    int n_points_;
    dtype* x_;
    dtype* y_;
    dtype* z_;
    dtype* w_;

    int n_boxes_;
    int* Bptr_;
    int* Bn_;
  } Boxes__gpu__t;

  typedef struct
  {
    int n_boxes_;
    int* L_;
    int* Ptr_;
  } UList_t;

  typedef struct
  {
    int n_boxes_;
    int* L_;
    int* Ptr_;
  } UList__gpu__t;


  struct FMMWrapper
  {
    AllNodes *AN;

    /* source boxes */
    Boxes_t S_h_;

    /* Target boxes */
    Boxes_t T_h_;

    /* Ulist */
    UList_t U_h_;

    /* GPU equivalent of the above */
    Boxes__gpu__t S_d_;
    Boxes__gpu__t T_d_;
    UList__gpu__t U_d_;

	int *tag_h_;

   int *depth_h_;
   int *child_h_;

		/*
		int *depth_src_h_;
		int *child_src_h_;
		int *depth_trg_h_;
		int *child_trg_h_;
		 */

    /* radius */
    real_t *radius_h_;
    real_t *center0_h_;
    real_t *center1_h_;
    real_t *center2_h_;

    /* GPU equivalent of the above */
		int *tag_d_;
    int *depth_d_;
    int *child_d_;
    dtype *radius_d_;
    dtype *center0_d_;
    dtype *center1_d_;
    dtype *center2_d_;

    /* SP[UC] */
    int SP_UC_size;
    int SP_UC_size_padded;
    real_t *SP_UC_h_;
    /* GPU equivalent of SP[UC] */
    dtype *SP_UC_d_;

    /* UC2UE matrix */
    int UC2UE_r;
    int UC2UE_r_padded;
    int UC2UE_c;
    real_t *UC2UE_h_;
    /* GPU equivalent of UC2UE matrix */
    dtype *UC2UE_d_;

    /* UE2UC matrix */
    int UE2UC_r;
    int UE2UC_r_padded;
    int UE2UC_c;
    real_t *UE2UC_h_;
    /* GPU equivalent of UE2UC matrix */
    dtype *UE2UC_d_;

    /* up_calc__gpu configuration variables */
    int tree_max_depth; /* maximum depht of the tree */
    int reduction_depth; /* where up_calc starts */
    int num_nodes_reduction; /* number of nodes at reduction_depth */
    int reduction_offset;
    // int num_non_leaf_nodes;

    /* temporary GPU up_calc variables */
		dtype *SRC_UPW_EQU_DEN_h_;
    dtype *SRC_UPW_EQU_DEN_d_;

    /* vlist arrays */
		int list_size;
    int *tlist_h_;
    int *vlist_h_;
    int *vlist_ptr_h_;

    int vlist_array_size;
    // int vlist_array_size_padded;

    /* GPU equivalent of above */
    int *tlist_d_;
    int *vlist_d_;
    int *vlist_ptr_d_;


    /* trans */
    int trans_arrays_num;
    dtype *vlist_trans_d_;
    /* src */
    dtype *vlist_src_d_;
    /* trg */
    dtype *vlist_trg_d_;

    /* ======================= */
    /* Rearranged */
    /* trans */
    int trans_arrays_num_pad;
    dtype *vlist_trans_d_re_;
    /* src */
    int num_nodes_pad;
    dtype *vlist_src_d_re_;
    /* trg */
    dtype *vlist_trg_d_re_;
    /* ======================= */

    /* Other structures */
    dtype *reg_den_d_;  /* needed for src */
    dtype *reg_den_ifft_d_;
    int reg_den_size;
    // int reg_den_size_padded;

    dtype *tt; /* needed for trans: size is of RP_n_ */

    /* RP */
    int RP_n_;
    /*
    dtype *RP_X_d_;
    dtype *RP_Y_d_;
    dtype *RP_Z_d_;
     */

    /* UE2DC -> Same as trans array */
    /* TRG_DWN_CHK_VAL */
    dtype *TRG_DWN_CHK_VAL_h_;
    dtype *TRG_DWN_CHK_VAL_d_;
    int SP_DC_n_;
    int SP_DC_n_padded_;


    /* DOWN_CALC */
    /* path2Node */
    int3 *path2Node_h_;
    int3 *path2Node_d_;

    /* parent */
    /* Don't need it - use children instead */
		/* Now I need it */
		int *parent_h_;
		int *parent_d_;

    /* trg_dwn_equ_den */
    int SP_DE_n_;
    int SP_DE_n_padded;
    dtype *TRG_DWN_EQU_DEN_d_;

    /* DC2DE_mat */
    int DC2DE_r;
    int DC2DE_r_padded;
    int DC2DE_c;
    real_t *DC2DE_h_;
    dtype *DC2DE_d_;

    /* DE2DC_mat */
    int DE2DC_r;
    int DE2DC_r_padded;
    int DE2DC_c;
    real_t* DE2DC_h_;
    dtype *DE2DC_d_;

    /* down_calc configuration parameters */
    int expansion_depth; /* where the first expansion occurs - 
                            depth 0 and 1 are skipped */
    int num_nodes_expansion; /* number of nodes at expansion depth */
    int expansion_offset; /* node ID of first node at expansion_depth */

    /* DOWN_CALC_LEAF */
    dtype *SP_DE_h_;
    dtype *SP_DE_d_;


		/* WLIST_CALC */
		int SP_UE_n_;
		int SP_UE_n_padded;
		dtype *SP_UE_h_;
		dtype *SP_UE_d_;

    /* Wlist */
    UList_t W_h_;
    /* GPU equivalent of the above */
    UList__gpu__t W_d_;

		int* srcNum_h_;
		int* srcNum_d_;


		/* XLIST_CALC */
		/* SP_DC_n_ is already defined */	
		/* SP_DC_n_padded is already defined */	
		dtype *SP_DC_h_;
		dtype *SP_DC_d_;

		UList_t X_h_;
		UList__gpu__t X_d_;

		int *trgNum_h_;
		int *trgNum_d_;
		
  };

#if defined (__cplusplus)
}
#endif

#endif
/* eof */
