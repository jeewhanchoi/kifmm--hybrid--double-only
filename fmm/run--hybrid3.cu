
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

int
run (FMMWrapper_t *F)
{
  AllNodes *All_N = F->AN;
  vector<NodeTree>& nodeVec = *All_N->N;

  struct stopwatch_t* timer = NULL;
  long double t_total;
  timer = stopwatch_create ();

    fprintf (stderr, "Performing Ulist on the GPU\n");
  stopwatch_start (timer);
    ulist_calc__gpu (F);
    up_calc__cpu (F);
    vlist_calc__cpu (F);
    cudaThreadSynchronize ();
		/* xfer up_calc result */
		/* first copy to SRC_UPW_EQU_DEN_h_ */
		for(int i = 0; i < nodeVec.size (); i++) {
			memcpy (&F->SRC_UPW_EQU_DEN_h_[i * F->UC2UE_r_padded], 
							&All_N->src_upw_equ_den[i * F->UC2UE_r], 
							F->UC2UE_r * sizeof (dtype));
		}
		/* first copy to SRC_UPW_EQU_DEN_h_ */
		for(int i = 0; i < nodeVec.size (); i++) {
			memcpy (&F->TRG_DWN_CHK_VAL_h_[i * F->SP_DC_n_padded_], 
							&All_N->trg_dwn_chk_val[i * F->SP_DC_n_], 
							F->SP_DC_n_ * sizeof (dtype));
		}
		/* DtoH copy of SRC_UPW_EQU_DEN */
		xfer__SRC_UPW_EQU_DEN__ (F);
		xfer__TRG_DWN_CHK_VAL__ (F);

		wlist_calc__gpu (F);
		xlist_calc__gpu (F);
    down_calc__gpu (F);
    copy_trg_val__gpu (F);
  t_total = stopwatch_stop (timer);
  
  fprintf(stderr, "=== Statistics ===\n");
  fprintf (stdout, "Code: %s\n", get_implementation_name ());
  fprintf (stdout, "Floating-point word size: %lu bytes\n", sizeof (real_t));
  fprintf (stderr, "== Timing for FMM3d ==\n");
  fprintf (stderr, "  ==> Total Execution Time: %Lg secs\n", t_total);

  stopwatch_destroy (timer);

  return 0;
}
/* ------------------------------------------------------------------------
 */
