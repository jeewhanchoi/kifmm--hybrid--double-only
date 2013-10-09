#ifndef __NODE_H__
#define __NODE_H__

#include <vector>
#include <fftw3.h>
#include "reals.h"
#include "reals_aligned.h"
#include "input.h"

#if defined (__cplusplus)
//extern "C" {
#endif

using namespace std;
/* ------------------------------------------------------------------------
 */
 
typedef struct {
  real_t *x;		// X coordinate of the points
  real_t *y;		// Y coordinate of the points
  real_t *z;		// Z coordinate of the points
  real_t *den_pot;	// Density/Potential of the points
  int num_pts;		// Number of points in the node
} Node;

#include "trans.h"

typedef struct {
  /* Pointer to re-arranged set of src and trg pts */
  Node *Ns;
  Node *Nt;

  /* Vector of all nodes */
  vector<NodeTree> *N;

  /* Original set of targets */
  real_t *tx_orig;
  real_t *ty_orig;
  real_t *tz_orig;
  real_t *pot_orig;

  /* Original set of sources */
  real_t *sx_orig;
  real_t *sy_orig;
  real_t *sz_orig;
  real_t *den_orig;

  /* Re-arranged set of targets after tree construction */
  real_t *tx;
  real_t *ty;
  real_t *tz;
  real_t *pot;
  
  /* Re-arranged set of sources after tree construction */
  real_t *sx;
  real_t *sy;
  real_t *sz;
  real_t *den;

  real_t *src_upw_equ_den;
  real_t *src_upw_chk_val;
  real_t *trg_dwn_equ_den;
  real_t *trg_dwn_chk_val;
  
  real_t *eff_den;
  real_t *eff_val;

  /* Structure for storing positions */ 
  Pos *SP;
  Pos *RP;

  Trans_matrix *TM;
  
  int *Tu;
  int *Tv;

  vector<int> *nodeLevel;
} AllNodes;

AllNodes* load_src_nodes (vector<NodeTree>& nodeVec);
AllNodes* load_trg_nodes (vector<NodeTree>& nodeVec);

typedef struct {
  int *node_list;	   // List of nodes in its list
  int num_listnodes;   // Number of nodes in the list
} List;

typedef struct {
   List* L; 
   int *nodelist_buffer; /* initialize in parallel later */
   int *counts;          /* temporary */
   int *offsets;
   int list_size;
} AllLists;

AllLists* load_lists (vector<NodeTree>& nodeVec);

void free_nodes(AllNodes *);
void free_lists(AllLists *);

double get_seconds(void);



int create (int N, char* distribution, AllNodes* All_N); 

int src_tree (int N, int pts_max, AllNodes* All_N);

int trg_tree (int N, AllNodes* All_N);

int plnDen2EffDeninit(int l, real_t* pln_den, real_t* eff_den, FFT_PLAN& forplan, AllNodes *All_N);

int plnDen2EffDen(int l, real_t* pln_den, real_t* eff_den, real_t* reg_den, real_t* tmp_den, FFT_PLAN& forplan, AllNodes *All_N);

int effVal2PlnValinit(int l, real_t* eff_val, real_t* pln_val, FFT_PLAN& invplan, AllNodes *All_N);

int effVal2PlnVal(int l, real_t* eff_val, real_t* pln_val, real_t* reg_val, FFT_PLAN invplan, AllNodes *All_N);
/* ------------------------------------------------------------------------
 */

#if defined (__cplusplus)
//}
#endif

#endif
