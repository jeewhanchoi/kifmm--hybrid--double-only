#ifndef __TRANS_H__
#define __TRANS_H__

#include "reals.h"

/*  UE = Upper Equivalent
  * UC = Upper Check
  * DE = Downward Equivalent
  * DC = Downward Check
  */
enum {UE=0, UC=1, DE=2, DC=3,}; 

typedef struct {
  real_t *x;
  real_t *y;
  real_t *z;
  int n;
} Pos;

typedef struct {
  int m, n;

  real_t *UC2UE;  
  real_t **UE2UC;
  real_t **UE2DC;
  real_t *DC2DE;  
  real_t **DE2DC;
}Trans_matrix;

#include "input.h"
#include "node.h"

int trans_setup (Pos *SP, Pos *RP);

int pln_size(int tp, Pos *SP);

int eff_data_size(int tp);

int compute_sampos(int np, real_t R, Pos *SP);

int compute_regpos(int np, real_t R, Pos *RP);

int compute_localpos (Point3 center, real_t radius, Node *t, Pos *SP);

int compute_UC2UE_mat (Trans_matrix *TM, Pos *SP);

int compute_UE2UC_mat (Index3 idx, Trans_matrix *TM, Pos *SP);

int compute_DC2DE_mat (Trans_matrix *TM, Pos *SP);

int compute_DE2DC_mat (Index3 idx, Trans_matrix *TM, Pos *SP);

int samDen2RegDen(const real_t* sam_den, real_t* reg_den);

int regVal2SamVal(const real_t* reg_val, real_t* sam_val);

/* ------------------------------------------------------------------------
 */

#endif
