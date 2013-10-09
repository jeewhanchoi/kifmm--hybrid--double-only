#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "node.h"
#include "evaluate.h"
#include "reals_aligned.h"
#include "util.h"

int 
get_input_ulist (AllNodes *All_N, int start_iter_trg, int end_iter_trg, int start_iter_src, int end_iter_src)
{
  int i;
  vector<NodeTree>& nodeVec = *All_N->N;

  /* Assign random values to all the target points */
  for (i = start_iter_trg; i < end_iter_trg; i++) {
    get_value (nodeVec[i].trgNum, All_N->tx + nodeVec[i].trgBeg, All_N->tx_orig, nodeVec[i].trgOwnVecIdxs);
    get_value (nodeVec[i].trgNum, All_N->ty + nodeVec[i].trgBeg, All_N->ty_orig, nodeVec[i].trgOwnVecIdxs);
    get_value (nodeVec[i].trgNum, All_N->tz + nodeVec[i].trgBeg, All_N->tz_orig, nodeVec[i].trgOwnVecIdxs);
    set_zero (nodeVec[i].trgNum, All_N->pot + nodeVec[i].trgBeg);
  }

  for (i = start_iter_trg; i < end_iter_trg; i++) {
    All_N->Nt[i].x = All_N->tx + nodeVec[i].trgBeg;
    All_N->Nt[i].y = All_N->ty + nodeVec[i].trgBeg;
    All_N->Nt[i].z = All_N->tz + nodeVec[i].trgBeg;
    All_N->Nt[i].den_pot = All_N->pot + nodeVec[i].trgBeg;
    All_N->Nt[i].num_pts = nodeVec[i].trgNum;
  }

  /* Assign random values to all the source points */ 
  for (i = start_iter_src; i < end_iter_src; i++) {
    get_value (nodeVec[i].srcNum, All_N->sx + nodeVec[i].srcBeg, All_N->sx_orig, nodeVec[i].srcOwnVecIdxs);
    get_value (nodeVec[i].srcNum, All_N->sy + nodeVec[i].srcBeg, All_N->sy_orig, nodeVec[i].srcOwnVecIdxs);
    get_value (nodeVec[i].srcNum, All_N->sz + nodeVec[i].srcBeg, All_N->sz_orig, nodeVec[i].srcOwnVecIdxs);
    get_value (nodeVec[i].srcNum, All_N->den + nodeVec[i].srcBeg, All_N->den_orig, nodeVec[i].srcOwnVecIdxs);
  }

  for (i = start_iter_src; i < end_iter_src; i++) {
    All_N->Ns[i].x = All_N->sx + nodeVec[i].srcBeg;
    All_N->Ns[i].y = All_N->sy + nodeVec[i].srcBeg;
    All_N->Ns[i].z = All_N->sz + nodeVec[i].srcBeg;
    All_N->Ns[i].den_pot = All_N->den + nodeVec[i].srcBeg;
    All_N->Ns[i].num_pts = nodeVec[i].srcNum;
  }

  return 1;
}

/* ------------------------------------------------------------------------
 */

void
get_input (AllNodes* All_N)
{
  int tid, nthreads;
  int num_nodes;
  int start_iter_T, end_iter_T;
  int start_iter_S, end_iter_S;
  int start_iter_Tv, end_iter_Tv;
  vector<NodeTree>& nodeVec = *All_N->N;
  num_nodes = nodeVec.size();

  if (getenv__numa()) {
    fprintf (stderr, "NUMA-aware allocation\n");
    #pragma omp parallel private (start_iter_S, end_iter_S, start_iter_T, end_iter_T, start_iter_Tv, end_iter_Tv, tid) 
    {
      nthreads = omp_get_num_threads ();
      tid = omp_get_thread_num ();
  
      start_iter_T = tid * num_nodes / nthreads; 
      end_iter_T   = (tid == nthreads-1) ? num_nodes : ((tid + 1) * num_nodes / nthreads);
      
      start_iter_S = tid * num_nodes / nthreads; 
      end_iter_S   = (tid == nthreads-1) ? num_nodes : ((tid + 1) * num_nodes / nthreads);
    
      get_input_ulist (All_N, start_iter_T, end_iter_T, start_iter_S, end_iter_S);

    }  /* end of parallel region */
  } 
  else {
    fprintf (stderr, "No NUMA-aware allocation\n");
    start_iter_T = 0;
    end_iter_T = nodeVec.size() ; 

    start_iter_S = 0;
    end_iter_S = nodeVec.size(); 

    get_input_ulist (All_N, start_iter_T, end_iter_T, start_iter_S, end_iter_S);
        
  } 
}
/* ------------------------------------------------------------------------
 */
void
work_division_U (AllNodes *All_N)
{
  int i, j;
  uint64_t sum;
  int nthreads;
  int num_nodes;
  int* Ti;
  vector<NodeTree>& nodeVec = *All_N->N;
  num_nodes = nodeVec.size();

  nthreads = get_num_threads();
  Ti = (int *) calloc ((nthreads + 1) , sizeof(int)); 
    
  /* Scan list once to determine offsets */
  sum = 0;
  for (i = 0; i < num_nodes; i++) {
    for (j = 0; j < nodeVec[i].Unodes.size(); j++) {
	    int src = nodeVec[i].Unodes[j];
      sum += All_N->Nt[i].num_pts * All_N->Ns[src].num_pts;
    }
  }

  uint64_t split = (sum/nthreads) + 1;
  uint64_t cutoff = split;
  int curr_thread = 1; 
  sum = 0;
  for (i = 0; i < num_nodes; i++) {
    for (j = 0; j < nodeVec[i].Unodes.size(); j++) {
	    int src = nodeVec[i].Unodes[j];
      sum += All_N->Nt[i].num_pts * All_N->Ns[src].num_pts;
      if (sum > cutoff) {
        Ti[curr_thread++] = i;
        cutoff += split;   
      }
    }
  }
  Ti[nthreads] = num_nodes;
  All_N->Tu = Ti;
}

/* ------------------------------------------------------------------------
 */
void
work_division_V (AllNodes *All_N)
{
  int i;
  int nthreads;
  int num_nodes;
  int* Ti;
  vector<NodeTree>& nodeVec = *All_N->N;
  num_nodes = nodeVec.size();
  
  nthreads = get_num_threads();
  Ti = (int *) calloc ((nthreads + 1) , sizeof(int)); 
  int* list_offsets = (int *) calloc ((num_nodes+1), sizeof(int));
    
  /* Scan list once to determine offsets */
  for (i = 0; i < num_nodes; i++) {
    int cnt = nodeVec[i].Vnodes.size();
    list_offsets[i+1] = list_offsets[i] + cnt;
  }

  int split = (list_offsets[num_nodes]/nthreads) + 1;
  int cutoff = split;
  int curr_thread = 1; 
  for (i = 0; i < num_nodes; i++) {
    if (list_offsets[i] > cutoff) {
      Ti[curr_thread++] = i;
      cutoff += split;   
    }
  }
  Ti[nthreads] = num_nodes;
  free (list_offsets);
  All_N->Tv = Ti;
}
/* ------------------------------------------------------------------------
 */
int
get_value (int n, real_t* x, real_t* xorig, vector<int>& val)
{
  int i, k;
  for (i = 0; i < n; i++) {
    k = val[i];
    x[i] = xorig[k];
  }

  return 0;
}

/* ------------------------------------------------------------------------
 */

int
set_value (int n, real_t* xorig, real_t* x, vector<int>& val)
{
  int i, k;
  for (i = 0; i < n; i++) {
    k = val[i];
    xorig[k] = x[i];
  }

  return 0;
}
/* ------------------------------------------------------------------------
 */

int
set_zero (int n, real_t* x)  
{
  int i;
  for (i = 0; i < n; i++) {
    x[i] = 0.0; 
  }

  return 0;
}

/* ------------------------------------------------------------------------
 */

int
set_rand (int n, real_t* x)  
{
  int i;
  for (i = 0; i < n; i++) {
    x[i] = drand48(); 
  }

  return 0;
}

/* ----------------------------------------------------------------------------------------------------------
 * eof
 */
