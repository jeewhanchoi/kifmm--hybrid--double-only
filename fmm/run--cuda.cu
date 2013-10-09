
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
#include <omp.h>

/* ------------------------------------------------------------------------
 */
/*const char *
get_implementation_name (void)
{
  return "cuda";
}
*/
/* ------------------------------------------------------------------------
 */

int
run (FMMWrapper_t *F)
{
  struct stopwatch_t* timer = NULL;
  long double t_up, t_u, t_v, t_w, t_x, t_down, t_total;
	long double t_u_start, t_w_start, t_x_start, t_down_start;

	long double t_v1_start, t_v2_start, t_v3_start, t_v4_start, t_copy_start;
	long double t_v1, t_v2, t_v3, t_v4, t_copy;
  timer = stopwatch_create ();

	#if MIN_DATA
		alloc__SOURCE_BOX__ (F);
		alloc__RADIUS__ (F);
		alloc__CENTER__ (F);
		alloc__SP_UC__ (F);
		alloc__UC2UE__ (F);
		alloc__DEPTH__ (F);
		alloc__SRC_UPW_EQU_DEN__ (F);
		alloc__CHILDREN__ (F);
		alloc__UE2UC__ (F);
		alloc__TAG__ (F);

		xfer__SOURCE_BOX__ (F);
		xfer__RADIUS__ (F);
		xfer__CENTER__ (F);
		xfer__SP_UC__ (F);
		xfer__UC2UE__ (F);
		xfer__DEPTH__ (F);
		xfer__CHILDREN__ (F);
		xfer__UE2UC__ (F);
		xfer__TAG__ (F);
	#endif
  fprintf (stderr, "Performing Up calculation ...\n"); 
  stopwatch_start (timer);
  up_calc__gpu (F);
  t_up = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Up.\n");
	#if MIN_DATA
		/* only SRC_UPW_EQU_DEN requires backing up */		
		xfer__SRC_UPW_EQU_DEN__back (F);

		free__SOURCE_BOX__ (F);
		free__RADIUS__ (F);
		free__CENTER__ (F);
		free__SP_UC__ (F);
		free__UC2UE__ (F);
		free__DEPTH__ (F);
		free__SRC_UPW_EQU_DEN__ (F);
		free__CHILDREN__ (F);
		free__UE2UC__ (F);
		free__TAG__ (F);
	#endif

	#if MIN_DATA
		alloc__SOURCE_BOX__ (F);
		alloc__TARGET_BOX__ (F);
		alloc__U_LIST__ (F);

		xfer__SOURCE_BOX__ (F);
		xfer__TARGET_BOX__ (F);
		xfer__U_LIST__ (F);
	#endif
  fprintf (stderr, "Performing U-list calculation (direct evaluation)...\n");
  t_u_start = stopwatch_elapsed (timer);
  ulist_calc__gpu (F);
	cudaThreadSynchronize ();
  t_u = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Ulist.\n");
	#if MIN_DATA
		/* backup TARGET */
		xfer__TARGET_BOX__back (F);

		free__SOURCE_BOX__ (F);
		free__TARGET_BOX__ (F);
		free__U_LIST__ (F);
	#endif

  fprintf (stderr, "Performing V-list calculation (pointwise multiply)...\n");
  // vlist_calc__gpu (F);

  AllNodes *All_N = F->AN;
	#if MIN_DATA
		alloc__DEPTH__ (F);
		alloc__SRC_UPW_EQU_DEN__ (F);
		alloc__REG_DEN__ (F);
		alloc__VLIST_SRC__ (F);

		xfer__DEPTH__ (F);
		xfer__SRC_UPW_EQU_DEN__ (F);
	#endif
  t_v1_start = stopwatch_elapsed (timer);
  compute_fft_src__gpu (F, All_N);
	cudaThreadSynchronize ();
  t_v1 = stopwatch_elapsed (timer);
	#if MIN_DATA
		// this is freed within compute_ifft_src__gpu
		// free__SRC_UPW_EQU_DEN__ (F);
		free__DEPTH__ (F);
		free__REG_DEN__ (F);	
	#endif

	#if MIN_DATA
		alloc__TT__ (F);
		alloc__VLIST_TRANS__ (F);
	#endif
  t_v2_start = stopwatch_elapsed (timer);
  compute_fft_trans__gpu (F, All_N);
	cudaThreadSynchronize ();
  t_v2 = stopwatch_elapsed (timer);
	#if MIN_DATA
		free__TT__ (F);
	#endif

	#if MIN_DATA
		alloc__VLIST_TRG__ (F);
		alloc__VLIST_TLIST__ (F);

		xfer__VLIST_TLIST__ (F);
	#endif
  t_v3_start = stopwatch_elapsed (timer);
  vlist_calc__gpu_ (F, All_N);
	cudaThreadSynchronize ();
  t_v3 = stopwatch_elapsed (timer);
	#if MIN_DATA
		free__VLIST_SRC__ (F);
		free__VLIST_TRANS__ (F);
		free__VLIST_TLIST__ (F);
	#endif

	#if MIN_DATA
		alloc__REG_DEN__ (F);
		// this is created within compute_ifft_trg__gpu
		// alloc__TRG_DWN_CHK_VAL__ (F);
	#endif
  t_v4_start = stopwatch_elapsed (timer);
  compute_ifft_trg__gpu (F, All_N);
	cudaThreadSynchronize ();
  t_v4 = stopwatch_elapsed (timer);
	#if MIN_DATA
		/* backup TRG_DWN_CHK_VAL */
		xfer__TRG_DWN_CHK_VAL__back (F);

		free__VLIST_TRG__ (F);
		free__REG_DEN__ (F);
		free__TRG_DWN_CHK_VAL__ (F);
	#endif

  // t_v = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Vlist.\n ");

  fprintf (stderr, "Performing W-list calculation...\n");
	#if MIN_DATA
		alloc__TAG__ (F);
		alloc__SRCNUM__ (F);
		alloc__CHILDREN__ (F);
		alloc__TARGET_BOX__ (F);
		alloc__SOURCE_BOX__ (F);
		alloc__W_LIST__ (F);
		alloc__SP_UE__ (F);
		alloc__RADIUS__ (F);
		alloc__CENTER__ (F);
		alloc__SRC_UPW_EQU_DEN__ (F);

		xfer__TAG__ (F);
		xfer__SRCNUM__ (F);
		xfer__CHILDREN__ (F);
		xfer__TARGET_BOX__ (F);
		xfer__SOURCE_BOX__ (F);
		xfer__SP_UE__ (F);
		xfer__W_LIST__ (F);
		xfer__RADIUS__ (F);
		xfer__CENTER__ (F);
		xfer__SRC_UPW_EQU_DEN__ (F);
	#endif
  t_w_start = stopwatch_elapsed (timer);
	wlist_calc__gpu (F);
	cudaThreadSynchronize ();
  t_w = stopwatch_elapsed (timer);
	#if MIN_DATA
		/* backup TARGET */
		xfer__TARGET_BOX__back (F);

		free__TAG__ (F);
		free__SRCNUM__ (F);
		free__CHILDREN__ (F);
		free__TARGET_BOX__ (F);
		free__SOURCE_BOX__ (F);
		free__W_LIST__ (F);
		free__SP_UE__ (F);
		free__RADIUS__ (F);
		free__CENTER__ (F);
		free__SRC_UPW_EQU_DEN__ (F);
	#endif
  fprintf (stderr, "Done Wlist.\n ");
 
  fprintf (stderr, "Performing X-list calculation...\n");
	#if MIN_DATA
		alloc__TAG__ (F);
		alloc__TRGNUM__ (F);
		alloc__CHILDREN__ (F);
		alloc__TARGET_BOX__ (F);
		alloc__SOURCE_BOX__ (F);
		alloc__X_LIST__ (F);
		alloc__SP_DC__ (F);
		alloc__RADIUS__ (F);
		alloc__CENTER__ (F);
		alloc__TRG_DWN_CHK_VAL__ (F);

		xfer__TAG__ (F);
		xfer__TRGNUM__ (F);
		xfer__CHILDREN__ (F);
		xfer__TARGET_BOX__ (F);
		xfer__SOURCE_BOX__ (F);
		xfer__X_LIST__ (F);
		xfer__SP_DC__ (F);
		xfer__RADIUS__ (F);
		xfer__CENTER__ (F);
		xfer__TRG_DWN_CHK_VAL__ (F);
	#endif
  t_x_start = stopwatch_elapsed (timer);
	xlist_calc__gpu (F);
	cudaThreadSynchronize ();
  t_x = stopwatch_elapsed (timer);
	#if MIN_DATA
		/* backup target box */
		xfer__TARGET_BOX__back (F);
		/* backup TRG_DWN_CHK_VAL */
		xfer__TRG_DWN_CHK_VAL__back (F);

		free__TAG__ (F);
		free__TRGNUM__ (F);
		free__CHILDREN__ (F);
		free__TARGET_BOX__ (F);
		free__SOURCE_BOX__ (F);
		free__X_LIST__ (F);
		free__SP_DC__ (F);
		free__RADIUS__ (F);
		free__CENTER__ (F);
		free__TRG_DWN_CHK_VAL__ (F);
	#endif	
  fprintf (stderr, "Done Xlist.\n ");

  fprintf (stderr, "Performing Down calculation ...\n");
	#if MIN_DATA
		alloc__TAG__ (F);
		alloc__TRG_DWN_CHK_VAL__ (F);
		alloc__DC2DE__ (F);
		alloc__TRG_DWN_EQU_DEN__ (F);
		alloc__DEPTH__ (F);
		alloc__PATH2NODE__ (F);
		alloc__PARENT__ (F);
		alloc__DE2DC__ (F);
		alloc__TARGET_BOX__ (F);
		alloc__SP_DE__ (F);
		alloc__RADIUS__ (F);
		alloc__CENTER__ (F);

		xfer__TAG__ (F);
		xfer__TRG_DWN_CHK_VAL__ (F);
		xfer__DC2DE__ (F);
		xfer__DEPTH__ (F);
		xfer__PATH2NODE__ (F);
		xfer__PARENT__ (F);
		xfer__DE2DC__ (F);
		xfer__TARGET_BOX__ (F);
		xfer__SP_DE__ (F);
		xfer__RADIUS__ (F);
		xfer__CENTER__ (F);
	#endif
  t_down_start = stopwatch_elapsed (timer);
  down_calc__gpu (F);
	cudaThreadSynchronize ();
  t_down = stopwatch_elapsed (timer);
	#if MIN_DATA
		free__TAG__ (F);
		free__TRG_DWN_CHK_VAL__ (F);
		free__DC2DE__ (F);
		free__TRG_DWN_EQU_DEN__ (F);
		free__DEPTH__ (F);
		free__PATH2NODE__ (F);
		free__PARENT__ (F);
		free__DE2DC__ (F);
		free__SP_DE__ (F);
		free__RADIUS__ (F);
		free__CENTER__ (F);

		// do this after copy_trg_val__gpu
		// free__TARGET_BOX__ (F);
	#endif
  fprintf (stderr, "Done Down.\n");

  t_copy_start = stopwatch_stop (timer);
  copy_trg_val__gpu (F);
  t_copy = stopwatch_stop (timer);

	t_v = (t_v1 - t_v1_start) + (t_v2 - t_v2_start) + (t_v3 - t_v3_start) + (t_v4 - t_v4_start);
	t_total = t_up + (t_u - t_u_start) + (t_v) + (t_w - t_w_start) + (t_x - t_x_start) + (t_down - t_down_start) + (t_copy - t_copy_start);

	#if MIN_DATA
		free__TARGET_BOX__ (F);
	#endif
  
  fprintf(stderr, "=== Statistics ===\n");
  fprintf (stdout, "Code: %s\n", get_implementation_name ());
  fprintf (stdout, "Floating-point word size: %lu bytes\n", sizeof (real_t));
  fprintf (stderr, "== Timing for FMM3d ==\n");
  fprintf (stderr, "  Up    : %Lg secs (%.1Lf%%)\n", t_up, t_up / t_total * 100);
  fprintf (stderr, "  U list: %Lg secs (%.1Lf%%)\n", (t_u - t_u_start), (t_u - t_u_start) / t_total * 100);
  fprintf (stderr, "  V list: %Lg secs (%.1Lf%%)\n", t_v, t_v / t_total * 100);
  fprintf (stderr, "  W list: %Lg secs (%.1Lf%%)\n", (t_w - t_w_start), (t_w - t_w_start) / t_total * 100);
  fprintf (stderr, "  X list: %Lg secs (%.1Lf%%)\n", (t_x - t_x_start), (t_x - t_x_start) / t_total * 100);
  fprintf (stderr, "  Down  : %Lg secs (%.1Lf%%)\n", (t_down - t_down_start), (t_down - t_down_start) / t_total * 100);
  fprintf (stderr, "  ==> Total Execution Time: %Lg secs\n", t_total);

  stopwatch_destroy (timer);

  return 0;
}
/* ------------------------------------------------------------------------
 */
