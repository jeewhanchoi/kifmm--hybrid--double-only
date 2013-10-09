#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "timing.h"
#include "node.h"
#include "reals.h"
#include "reals_aligned.h"
#include "evaluate.h"
#include "util.h"
#include "trans.h"
#include "input.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static
void
usage__ (const char* use)
{
  fprintf (stderr, "usage: %s <N> <distribution> <pts/box> \n", use);
}

/* ----------------------------------------------------------------------------------------------------------
*/

double get_seconds () {
  struct timeval tv;
  double t;
  gettimeofday(&tv, NULL);
  t = (double)tv.tv_sec + 
    ((double)tv.tv_usec)/(double)1000000.0;
  return t;
}

/* ----------------------------------------------------------------------------------------------------------
*/
/**
 *  \brief Returns an estimate of the number of flops required to
 *  perform the U-list computation using the naive Laplacian U-list.
 */
static
long double
estimate_flops_U__ (AllNodes *All_N)
{
  assert (All_N);
  vector<NodeTree>& nodeVec = *All_N->N;
  Node *Nt = All_N->Nt;
  Node *Ns = All_N->Ns;

  long double flop_count = 0.0;
  int tgt_box_id;

  for (tgt_box_id = 0; tgt_box_id < nodeVec.size(); ++tgt_box_id) {
    int k;
    for (k = 0; k < nodeVec[tgt_box_id].Unodes.size(); ++k) {
      int src_box_id = nodeVec[tgt_box_id].Unodes[k];
      flop_count += 11.0 * Nt[tgt_box_id].num_pts * Ns[src_box_id].num_pts;
    }
  }
  return flop_count;
}

/**
 *  \brief Returns an estimate of the number of flops required to
 *  perform the V-list computation using the naive V-list.
 */
static
long double
estimate_flops_V__ (AllNodes *All_N)
{
  assert (All_N);
  vector<NodeTree>& nodeVec = *All_N->N;
  int trg_nodes = nodeVec.size();
  int size = eff_data_size (UE);
 
  long double flop_count = 0.0;
  int tgt_box_id;
  for (tgt_box_id = 0; tgt_box_id < trg_nodes; ++tgt_box_id) {
    flop_count += 4.0 * size * nodeVec[tgt_box_id].Vnodes.size();
  }
  return flop_count;
}

/**
 *  \brief Returns an estimate of the maximum number of bytes that
 *  need to be loaded to perform the U-list computation using the
 *  given U-list.
 */
static
long double
estimate_bytes_U__ (AllNodes *All_N)
{
  assert (All_N);
  vector<NodeTree>& nodeVec = *All_N->N;
  Node *Nt = All_N->Nt;
  Node *Ns = All_N->Ns;

  long double byte_count = 0.0;
  int tgt_box_id;

  for (tgt_box_id = 0; tgt_box_id < nodeVec.size(); ++tgt_box_id) {
    int k;
    byte_count += 4.0 * sizeof (real_t) * Nt[tgt_box_id].num_pts;
    for (k = 0; k < nodeVec[tgt_box_id].Unodes.size(); ++k) {
      int src_box_id = nodeVec[tgt_box_id].Unodes[k];
      byte_count += 4.0 * sizeof (real_t) * Ns[src_box_id].num_pts;
    }
  }
  return byte_count;
}

/**
 *  \brief Returns an estimate of the maximum number of bytes that
 *  need to be loaded to perform the V-list computation using the
 *  given V-list.
 */
static
long double
estimate_bytes_V__ (AllNodes *All_N)
{
  assert (All_N);
  vector<NodeTree>& nodeVec = *All_N->N;
  int trg_nodes = nodeVec.size();
  int size = eff_data_size (UE);
  
  long double byte_count = 0.0;
  int tgt_box_id;
  for (tgt_box_id = 0; tgt_box_id < trg_nodes; ++tgt_box_id) {
    byte_count += sizeof (real_t) * size;
    byte_count += 2.0 * sizeof (real_t) * size * nodeVec[tgt_box_id].Vnodes.size();
  }
  return byte_count;
}

/* ----------------------------------------------------------------------------------------------------------
*/

int
validate (int num_trg, int num_src, int num_chk, AllNodes *All_N, real_t &rerr)
{
  int trgDOF = 1;
  int srcDOF = 1;

  /* Randomly select points to check */
  int chk_id [num_chk];
  for (int k = 0; k < num_chk; k++) {
	  chk_id[k] = int( floor(drand48()*num_trg) );	 
    assert (chk_id[k]>=0 && chk_id[k]<num_trg);
  }
  real_t* chkPosX = reals_alloc__aligned (num_chk);
  real_t* chkPosY = reals_alloc__aligned (num_chk);
  real_t* chkPosZ = reals_alloc__aligned (num_chk);
  real_t* chkVal = reals_alloc__aligned (num_chk * trgDOF);
  real_t* chkPot = reals_alloc__aligned (num_chk * trgDOF);
  for (int k = 0; k < num_chk; k++) {
	  chkPosX[k] = All_N->tx_orig[chk_id[k]];
	  chkPosY[k] = All_N->ty_orig[chk_id[k]];
	  chkPosZ[k] = All_N->tz_orig[chk_id[k]];
	  for (int i = 0; i < trgDOF; i++)		
      chkVal[k*trgDOF+i] = All_N->pot_orig[chk_id[k]*trgDOF+i];
  }
  
  /* Compute direct N^2 on selected points */
  Node* src;
  src = (Node *) _mm_malloc(sizeof(Node), IDEAL_ALIGNMENT);
  src->x = All_N->sx_orig;
  src->y = All_N->sy_orig;
  src->z = All_N->sz_orig;
  src->den_pot = All_N->den_orig;
  src->num_pts = num_src;

  real_t* inter = reals_alloc__aligned (trgDOF * num_src * srcDOF);

  for (int i = 0; i < num_chk; i++) {
    Node* chk_pos;
    chk_pos = (Node *) _mm_malloc(sizeof(Node), IDEAL_ALIGNMENT);
    chk_pos->x = &chkPosX[i]; 
    chk_pos->y = &chkPosY[i]; 
    chk_pos->z = &chkPosZ[i]; 
    chk_pos->den_pot = &chkPot[i]; 
    chk_pos->num_pts = trgDOF;
	  ulist__direct_evaluation(*chk_pos, *src);
  }
  
  /* Compute relative error */
  for (int k = 0; k < num_chk; k++)
	  for (int i = 0; i < trgDOF; i++)
		  chkPot[k*trgDOF+i] -= chkVal[k*trgDOF+i];
  
  real_t vn = 0;  
  real_t en = 0;
  for (int k = 0; k < num_chk; k++)
	  for (int i = 0; i < trgDOF; i++) {
		  vn += chkVal[k*trgDOF+i] * chkVal[k*trgDOF+i];
		  en += chkPot[k*trgDOF+i] * chkPot[k*trgDOF+i];
	  }
  vn = sqrt(vn);
  en = sqrt(en);
  
  rerr = en/vn; 
  
  return 0;
}

/* ----------------------------------------------------------------------------------------------------------
*/

int main (int argc, char* argv[])
{
  char* distribution;
  int i;
  int num_pts;
  int num_chk;
  int pts_max;
  int src_node_count, trg_node_count;  
  real_t rerr;

  vector<NodeTree> nodeVec;
  vector<int> nodeLevel;
  AllNodes* All_N;

  Node *Ns, *Nt;
  FMMWrapper_t* F;

  /* timing */
  struct stopwatch_t* timer = NULL;
  long double t_wu, t_wv;
  long double num_flops_U, num_flops_V;
  long double num_bytes_U, num_bytes_V;

  if (argc != 4) {
    usage__ (argv[0]);
    return -1;
  }

  /** Command line input */ 
  num_pts = atoi (argv[1]);
  distribution = argv[2];
  pts_max = atoi (argv[3]);

  srand48(26);

  /** Allocate memory for structure AllNodes for all nodes in the tree */
  All_N = (AllNodes *) malloc (sizeof (AllNodes));
  assert (All_N != NULL);
  
  All_N->tx_orig = (real_t *) reals_alloc__aligned (num_pts);
  All_N->ty_orig = (real_t *) reals_alloc__aligned (num_pts);
  All_N->tz_orig = (real_t *) reals_alloc__aligned (num_pts);
  All_N->pot_orig = (real_t *) reals_alloc__aligned (num_pts);

  All_N->sx_orig = (real_t *) reals_alloc__aligned (num_pts);
  All_N->sy_orig = (real_t *) reals_alloc__aligned (num_pts);
  All_N->sz_orig = (real_t *) reals_alloc__aligned (num_pts);
  All_N->den_orig = (real_t *) reals_alloc__aligned (num_pts);
  
  All_N->N = &nodeVec;
  All_N->nodeLevel = &nodeLevel;

  /** Generate source and target points according to the distribution */
  create (num_pts, distribution, All_N);
   
  for (int i = 0; i < num_pts; i++)
    All_N->den_orig[i] = drand48();
 
  /** Construct octree for sources */
  src_tree (num_pts, pts_max, All_N);

  /** Construct octree for targets */
  trg_tree (num_pts, All_N);

  /* Initialize RNG with seed */
//  srand48(232323);

  stopwatch_init ();
  timer = stopwatch_create ();

  fprintf (stderr, "Allocating memory for re-arranged source and target points...\n");
  All_N->tx = (real_t *) reals_alloc__aligned (num_pts);
  All_N->ty = (real_t *) reals_alloc__aligned (num_pts);
  All_N->tz = (real_t *) reals_alloc__aligned (num_pts);
  All_N->pot = (real_t *) reals_alloc__aligned (num_pts);

  All_N->sx = (real_t *) reals_alloc__aligned (num_pts);
  All_N->sy = (real_t *) reals_alloc__aligned (num_pts);
  All_N->sz = (real_t *) reals_alloc__aligned (num_pts);
  All_N->den = (real_t *) reals_alloc__aligned (num_pts);

  Ns = (Node *) _mm_malloc(nodeVec.size() * sizeof(Node), IDEAL_ALIGNMENT);
  Nt = (Node *) _mm_malloc(nodeVec.size() * sizeof(Node), IDEAL_ALIGNMENT);
  
  All_N->Ns = Ns; 
  All_N->Nt = Nt;
 
  fprintf (stderr, "Performing initialization...\n");
  get_input (All_N);

  /* Translations Setup */
  fprintf (stderr, "Setup for translations....\n");
  Pos *SP;
  SP = (Pos*) malloc (sizeof(Pos) * 4); 
  
  Pos *RP;
  RP = (Pos*) malloc (sizeof(Pos)); 

  trans_setup (SP, RP);
  All_N->SP = SP;
  All_N->RP = RP;

  Trans_matrix *TM;
  TM = (Trans_matrix*) malloc (sizeof(Trans_matrix)); 
  All_N->TM = TM;
  
  fprintf (stderr, "Allocating memory for translations...\n");
  All_N->src_upw_equ_den = (real_t *) reals_alloc__aligned (nodeVec.size() * pln_size(UE, All_N->SP));
  All_N->eff_den = (real_t *) reals_alloc__aligned (nodeVec.size() * eff_data_size(UE));
  All_N->trg_dwn_equ_den = (real_t *) reals_alloc__aligned (nodeVec.size() * pln_size(DE, All_N->SP));
  All_N->trg_dwn_chk_val = (real_t *) reals_alloc__aligned (nodeVec.size() * pln_size(DC, All_N->SP));
  All_N->eff_val = (real_t *) reals_alloc__aligned (nodeVec.size() * eff_data_size(DC));
  

  fprintf (stderr, "Performing U-list work load division among the threads...\n");
  work_division_U (All_N);

  fprintf (stderr, "Performing V-list work load division among the threads...\n");
  work_division_V (All_N);
 
  fprintf (stderr, "Preprocessing...\n");
  F = preproc (All_N);  
  
  fprintf(stderr, "Finished setup.\n");

  fprintf (stderr, "Evaluation...\n");
  run (F);
  
  /* Verify that FMM returns the correct results */
  num_chk = getenv__validate();
  validate (num_pts, num_pts, num_chk, All_N, rerr);

  num_flops_U = estimate_flops_U__ (All_N);
  num_flops_V = estimate_flops_V__ (All_N);

  num_bytes_U = estimate_bytes_U__ (All_N);
  num_bytes_V = estimate_bytes_V__ (All_N);
  

  fprintf (stdout, "Inputs: %d | %s | %d \n",
	   num_pts,
	   distribution,
     pts_max);
  fprintf (stdout, "Estimated no. of flops (Ulist): %Lg\n",
	   num_flops_U);
  fprintf (stdout, "Estimated no. of flops (Vlist): %Lg\n",
	   num_flops_V);
  fprintf (stdout, "Lower-bound on maximum memory traffic (Ulist): %Lg GB\n",
	   1e-9 * num_bytes_U);
  fprintf (stdout, "Lower-bound on maximum memory traffic (Vlist): %Lg GB\n",
	   1e-9 * num_bytes_V);
//  fprintf (stdout, "Performance (estimated): %.2Lf Gflop/s\n",
//	   1e-9 * (num_flops_U + num_flops_V) / t_uv);
  fprintf (stdout, "Lower-bound on average intensity (Ulist): %.2Lf flops / byte\n",
	   num_flops_U / num_bytes_U);
  fprintf (stdout, "Lower-bound on average intensity (Vlist): %.2Lf flops / byte\n",
	   num_flops_V / num_bytes_V);
  fprintf (stderr, "Relative error: %e\n", rerr); 

  stopwatch_destroy (timer);
  
  return 0;
}

/* ----------------------------------------------------------------------------------------------------------
 * eof
 */
