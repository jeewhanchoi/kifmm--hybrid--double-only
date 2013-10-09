

/* up_calc requires:
 * __SOURCE_BOX__
 * __RADIUS__
 * __CENTER__
 * __SP_UC__
 * __UC2UE__
 * __DEPTH__
 * __SRC_UPW_EQU_DEN__
 * 
 * __CHILDREN__
 * __SRC_UPW_EQU_DEN__ <== again
 * __UE2UC__
 * __UC2UE__ <== again
 *
 * __CHILDREN__ <== again
 * __SRC_UPW_EQU_DEN__ <== again <==again
 * __UE2UC__ <== again
 * __UC2UE__ <== again <== again
 */


/* ulist_calc requires:
 * __TARGET_BOX__
 * __SOURCE_BOX__
 * __U_LIST__
 */


/* vlist_calc requires: 
 * __DEPTH__
 * __SRC_UPW_EQU_DEN__
 * __REG_DEN__
 * __VLIST_SRC__
 * __TT__
 * __VLIST_TRANS__
 * __VLIST_TRG__
 * __VLIST_TLIST__
 * __REG_DEN_IFFT__
 * __TRG_DWN_CHK_VAL__
 */

/* down_calc requires 
 * __CHILDREN__
 * __PATH2NODE__
 * __TRG_DWN_CHK_VAL__
 * __TRG_DWN_EQU_DEN__
 * __DC2DE__
 * __DE2DC__
 *
 * __TARGET_BOX__
 * __SP_DE__
 * __TRG_DWN_EQU_DEN__
 * __RADIUS__
 * __CENTER__
 */
#define MIN_DATA 0

#if MIN_DATA
#define __SOURCE_BOX__ 0
#define __TARGET_BOX__ 0
#define __U_LIST__ 0
#define __TAG__ 0
#define __DEPTH__ 0
#define __CHILDREN__ 0
#define __RADIUS__ 0
#define __CENTER__ 0
#define __SP_UC__ 0
#define __UC2UE__  0
#define __UE2UC__ 0
#define __SRC_UPW_EQU_DEN__ 0
#define __VLIST_SRC__ 0
#define __REG_DEN__ 0
#define __TT__ 0
#define __VLIST_TRANS__ 0
#define __VLIST_TRG__ 0
#define __VLIST_TLIST__ 0
#define __REG_DEN_IFFT__ 0
#define __TRG_DWN_CHK_VAL__ 0
#define __PATH2NODE__ 0
#define __TRG_DWN_EQU_DEN__ 0
#define __PARENT__ 0
#define __DC2DE__ 0
#define __DE2DC__ 0
#define __SP_DE__ 0
#define __SP_UE__ 0
#define __W_LIST__ 0
#define __SRCNUM__ 0
#define __SP_DC__ 0
#define __X_LIST__ 0
#define __TRGNUM__ 0

#else
#define __SOURCE_BOX__ 1
#define __TARGET_BOX__ 1
#define __U_LIST__ 1
#define __TAG__ 1
#define __DEPTH__ 1
#define __CHILDREN__ 1
#define __RADIUS__ 1
#define __CENTER__ 1
#define __SP_UC__ 1
#define __UC2UE__  1
#define __UE2UC__ 1
#define __SRC_UPW_EQU_DEN__ 1
#define __VLIST_SRC__ 1
#define __REG_DEN__ 1
#define __TT__ 1
#define __VLIST_TRANS__ 1
#define __VLIST_TRG__ 1
#define __VLIST_TLIST__ 1
#define __REG_DEN_IFFT__ 0 /* no longer used */
#define __TRG_DWN_CHK_VAL__ 1
#define __PATH2NODE__ 1
#define __TRG_DWN_EQU_DEN__ 1
#define __PARENT__ 1
#define __DC2DE__ 1
#define __DE2DC__ 1
#define __SP_DE__ 1
#define __SP_UE__ 1
#define __W_LIST__ 1
#define __SRCNUM__ 1
#define __SP_DC__ 1
#define __X_LIST__ 1
#define __TRGNUM__ 1
#endif



#if NP_3
/* up_eval__gpu */
/* This should be the same as SP[UC].n */
#define BLK_SIZE_UP 112

/* up_eval__gpu_reduction */
/* this should be 8 * uc2ue_r_padded (SP[UE]) */
#define UC2UE_R_PADDED 256

/* size should be 8x ue2uc_r_padded (SP[UC]) */
#define UE2UC_R_PADDED 896 

/* compute_fft_src__gpu_eval */
/* size should be RP->n =(2np)^3 */
#define RP_N 224

/* incorrectly named */
/* UC2UE_R_PADDED (fake) <= 8 * UC2UE_R_PADDED (real) */
#define UC2UE_R (UC2UE_R_PADDED/8)
#define NP_CUBED_POWER_OF_2 32

/* compute_ifft_trg__gpu_regVal2SamVal */
/* this should be qual to sp_dc_n_padded */
#define SP_DC_N 32

/* down_eval__gpu */
#define SP_DE_N_PADDED_8 256

/* down_eval__gpu_leaf */
/* should be equal to SP[DE].n padded*/
#define SL_POS_SIZE 32

#endif



#if NP_4
/* up_eval__gpu */
/* This should be the same as SP[UC].n */
#define BLK_SIZE_UP 160

/* up_eval__gpu_reduction */
/* this should be 8 * uc2ue_r_padded (SP[UE]) */
#define UC2UE_R_PADDED 512

/* size should be 8x ue2uc_r_padded (SP[UC]) */
#define UE2UC_R_PADDED 1280

/* compute_fft_src__gpu_eval */
/* size should be RP->n =(2np)^3 */
#define RP_N 512

/* incorrectly named */
/* UC2UE_R_PADDED (fake) <= 8 * UC2UE_R_PADDED (real) */
#define UC2UE_R (UC2UE_R_PADDED/8)
#define NP_CUBED_POWER_OF_2 64

/* compute_ifft_trg__gpu_regVal2SamVal */
/* this should be qual to sp_dc_n_padded */
#define SP_DC_N 64

/* down_eval__gpu */
#define SP_DE_N_PADDED_8 512

/* down_eval__gpu_leaf */
/* should be equal to SP[DE].n padded*/
#define SL_POS_SIZE 64

#endif

#if NP_6
/* up_eval__gpu */
/* This should be the same as SP[UC].n */
#define BLK_SIZE_UP 304

/* up_eval__gpu_reduction */
/* this should be 8 * uc2ue_r_padded (SP[UE]) */
#define UC2UE_R_PADDED 1280
/* size should be 8x ue2uc_r_padded (SP[UC]) */
#define UE2UC_R_PADDED 2432

/* compute_fft_src__gpu_eval */
/* size should be RP->n =(2np)^3 */
#define RP_N 1728

/* incorrectly named */
/* UC2UE_R_PADDED (fake) <= 8 * UC2UE_R_PADDED (real) */
#define UC2UE_R (UC2UE_R_PADDED/8)
#define NP_CUBED_POWER_OF_2 256

/* compute_ifft_trg__gpu_regVal2SamVal */
/* this should be qual to sp_dc_n_padded */
#define SP_DC_N 160

/* down_eval__gpu */
#define SP_DE_N_PADDED_8 1280

/* down_eval__gpu_leaf */
/* should be equal to SP[DE].n padded*/
#define SL_POS_SIZE 160
#endif

#if NP_8
/* up_eval__gpu */
/* This should be the same as SP[UC].n */
#define BLK_SIZE_UP 496

/* up_eval__gpu_reduction */
/* this should be 8 * uc2ue_r_padded (SP[UE]) */
#define UC2UE_R_PADDED 2432

/* size should be 8x ue2uc_r_padded (SP[UC]) */
#define UE2UC_R_PADDED 3968

/* compute_fft_src__gpu_eval */
/* size should be RP->n =(2np)^3 */
#define RP_N 4096

/* incorrectly named */
/* UC2UE_R_PADDED (fake) <= 8 * UC2UE_R_PADDED (real) */
#define UC2UE_R (UC2UE_R_PADDED/8)
#define NP_CUBED_POWER_OF_2 512

/* compute_ifft_trg__gpu_regVal2SamVal */
/* this should be qual to sp_dc_n_padded */
#define SP_DC_N 304

/* down_eval__gpu */
#define SP_DE_N_PADDED_8 2432

/* down_eval__gpu_leaf */
/* should be equal to SP[DE].n padded*/
#define SL_POS_SIZE 304

#endif



