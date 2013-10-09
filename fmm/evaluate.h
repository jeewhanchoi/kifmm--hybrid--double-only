#ifndef __EVALUATE_H__
#define __EVALUATE_H__

/* ------------------------------------------------------------------------
 */

#include <math.h>
#include <node.h>

#define OOFP_R  (1.0/(4.0 * M_PI))

//#if defined (__cplusplus)
//extern "C" {
//#endif

typedef struct FMMWrapper FMMWrapper_t;

FMMWrapper_t* preproc (AllNodes *All_N);
void work_division_U (AllNodes *All_N);
void work_division_V (AllNodes *All_N);
const char* get_implementation_name (void);
int get_num_threads (void);
int run (FMMWrapper_t *F);
int get_value (int n, real_t* x, real_t* xorig, vector<int>& val);
int set_value (int n, real_t* xorig, real_t* x, vector<int>& val);
int set_zero (int n, real_t* x);
int set_rand (int n, real_t* x);
void get_input (AllNodes* All_N);

int ulist__direct_evaluation (Node trg, Node src);
int pointwise_mult (int n, FFT_COMPLEX* x, int ix, FFT_COMPLEX* y, int iy, FFT_COMPLEX* z, int iz);
int kernel (int sn, int tn, real_t* x1, real_t* x2, real_t* x3, real_t* y1, real_t* y2, real_t* y3, real_t* mat); 

int up_calc__cpu (FMMWrapper_t* F);
int ulist_calc__cpu (FMMWrapper_t* F);
int vlist_calc__cpu (FMMWrapper_t* F);
int wlist_calc__cpu (FMMWrapper_t* F);
int xlist_calc__cpu (FMMWrapper_t* F);
int down_calc__cpu (FMMWrapper_t* F);
int copy_trg_val__cpu (FMMWrapper_t* F);

int up_calc__gpu (FMMWrapper_t* F);
int ulist_calc__gpu (FMMWrapper_t* F);
int vlist_calc__gpu (FMMWrapper_t* F);
	void compute_fft_src__gpu (FMMWrapper_t *f, AllNodes *All_N);
  void compute_fft_trans__gpu (FMMWrapper_t *f, AllNodes *All_N);
	void vlist_calc__gpu_ (FMMWrapper_t *f, AllNodes *All_N);
	void compute_ifft_trg__gpu (FMMWrapper_t *f, AllNodes *All_N);



int wlist_calc__gpu (FMMWrapper_t* F);
int xlist_calc__gpu (FMMWrapper_t* F);
int down_calc__gpu (FMMWrapper_t* F);
int d2d__gpu (FMMWrapper_t* F);
int d2t__gpu (FMMWrapper_t* F);
int copy_trg_val__gpu (FMMWrapper_t* F);



void alloc__SOURCE_BOX__ (FMMWrapper_t* f);
void alloc__TARGET_BOX__ (FMMWrapper_t* f);
void alloc__U_LIST__ (FMMWrapper_t* f);
void alloc__TAG__ (FMMWrapper_t* f);
void alloc__DEPTH__ (FMMWrapper_t* f);
void alloc__CHILDREN__ (FMMWrapper_t* f);
void alloc__RADIUS__ (FMMWrapper_t* f);
void alloc__CENTER__ (FMMWrapper_t* f);
void alloc__SP_UC__ (FMMWrapper_t* f);
void alloc__UC2UE__ (FMMWrapper_t* f);
void alloc__UE2UC__ (FMMWrapper_t* f);
void alloc__SRC_UPW_EQU_DEN__ (FMMWrapper_t* f);
void alloc__VLIST_SRC__ (FMMWrapper_t* f);
void alloc__REG_DEN__ (FMMWrapper_t* f);
void alloc__TT__ (FMMWrapper_t* f);
void alloc__VLIST_TRANS__ (FMMWrapper_t* f);
void alloc__VLIST_TRG__ (FMMWrapper_t* f);
void alloc__VLIST_TLIST__ (FMMWrapper_t* f);
void alloc__TRG_DWN_CHK_VAL__ (FMMWrapper_t* f);
void alloc__PATH2NODE__ (FMMWrapper_t* f);
void alloc__PARENT__ (FMMWrapper_t* f);
void alloc__TRG_DWN_EQU_DEN__ (FMMWrapper_t* f);
void alloc__DC2DE__ (FMMWrapper_t* f);
void alloc__DE2DC__ (FMMWrapper_t* f);
void alloc__SP_DE__ (FMMWrapper_t* f);
void alloc__SP_UE__ (FMMWrapper_t* f);
void alloc__W_LIST__ (FMMWrapper_t* f);
void alloc__SRCNUM__ (FMMWrapper_t* f);
void alloc__SP_DC__ (FMMWrapper_t* f);
void alloc__X_LIST__ (FMMWrapper_t* f);
void alloc__TRGNUM__ (FMMWrapper_t* f);

void xfer__SOURCE_BOX__ (FMMWrapper_t* f);
void xfer__TARGET_BOX__ (FMMWrapper_t* f);
void xfer__U_LIST__ (FMMWrapper_t* f);
void xfer__TAG__ (FMMWrapper_t* f);
void xfer__DEPTH__ (FMMWrapper_t* f);
void xfer__CHILDREN__ (FMMWrapper_t* f);
void xfer__RADIUS__ (FMMWrapper_t* f);
void xfer__CENTER__ (FMMWrapper_t* f);
void xfer__SP_UC__ (FMMWrapper_t* f);
void xfer__UC2UE__ (FMMWrapper_t* f);
void xfer__UE2UC__ (FMMWrapper_t* f);
void xfer__SRC_UPW_EQU_DEN__ (FMMWrapper_t* f);
// void alloc__VLIST_SRC__ (FMMWrapper_t* f);
// void alloc__REG_DEN__ (FMMWrapper_t* f);
// void alloc__TT__ (FMMWrapper_t* f);
// void alloc__VLIST_TRANS__ (FMMWrapper_t* f);
// void alloc__VLIST_TRG__ (FMMWrapper_t* f);
void xfer__VLIST_TLIST__ (FMMWrapper_t* f);
void xfer__TRG_DWN_CHK_VAL__ (FMMWrapper_t* f);
void xfer__PATH2NODE__ (FMMWrapper_t* f);
void xfer__PARENT__ (FMMWrapper_t* f);
// void alloc__TRG_DWN_EQU_DEN__ (FMMWrapper_t* f);
void xfer__DC2DE__ (FMMWrapper_t* f);
void xfer__DE2DC__ (FMMWrapper_t* f);
void xfer__SP_DE__ (FMMWrapper_t* f);
void xfer__SP_UE__ (FMMWrapper_t* f);
void xfer__W_LIST__ (FMMWrapper_t* f);
void xfer__SRCNUM__ (FMMWrapper_t* f);
void xfer__SP_DC__ (FMMWrapper_t* f);
void xfer__X_LIST__ (FMMWrapper_t* f);
void xfer__TRGNUM__ (FMMWrapper_t* f);



void free__SOURCE_BOX__ (FMMWrapper_t* f);
void free__TARGET_BOX__ (FMMWrapper_t* f);
void free__U_LIST__ (FMMWrapper_t* f);
void free__TAG__ (FMMWrapper_t* f);
void free__DEPTH__ (FMMWrapper_t* f);
void free__CHILDREN__ (FMMWrapper_t* f);
void free__RADIUS__ (FMMWrapper_t* f);
void free__CENTER__ (FMMWrapper_t* f);
void free__SP_UC__ (FMMWrapper_t* f);
void free__UC2UE__ (FMMWrapper_t* f);
void free__UE2UC__ (FMMWrapper_t* f);
void free__SRC_UPW_EQU_DEN__ (FMMWrapper_t* f);
void free__VLIST_SRC__ (FMMWrapper_t* f);
void free__REG_DEN__ (FMMWrapper_t* f);
void free__TT__ (FMMWrapper_t* f);
void free__VLIST_TRANS__ (FMMWrapper_t* f);
void free__VLIST_TRG__ (FMMWrapper_t* f);
void free__VLIST_TLIST__ (FMMWrapper_t* f);
void free__TRG_DWN_CHK_VAL__ (FMMWrapper_t* f);
void free__PATH2NODE__ (FMMWrapper_t* f);
void free__PARENT__ (FMMWrapper_t* f);
void free__TRG_DWN_EQU_DEN__ (FMMWrapper_t* f);
void free__DC2DE__ (FMMWrapper_t* f);
void free__DE2DC__ (FMMWrapper_t* f);
void free__SP_DE__ (FMMWrapper_t* f);
void free__SP_UE__ (FMMWrapper_t* f);
void free__W_LIST__ (FMMWrapper_t* f);
void free__SRCNUM__ (FMMWrapper_t* f);
void free__SP_DC__ (FMMWrapper_t* f);
void free__X_LIST__ (FMMWrapper_t* f);
void free__TRGNUM__ (FMMWrapper_t* f);


void xfer__SRC_UPW_EQU_DEN__back (FMMWrapper_t* f);
void xfer__TRG_DWN_CHK_VAL__back (FMMWrapper_t* f);
void xfer__TARGET_BOX__back (FMMWrapper_t* f);

//#if defined (__cplusplus)
//}
//#endif

/* ------------------------------------------------------------------------
 */

#endif
