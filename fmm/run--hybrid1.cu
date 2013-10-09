
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
  struct stopwatch_t* timer = NULL;
  long double t_total;
  timer = stopwatch_create ();

    fprintf (stderr, "Performing Ulist on the GPU\n");
  stopwatch_start (timer);
    ulist_calc__gpu (F);
    up_calc__cpu (F);
    vlist_calc__cpu (F);
		// wlist_calc__cpu (F);
		// xlist_calc__cpu (F);
    down_calc__cpu (F);
    cudaThreadSynchronize ();
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
