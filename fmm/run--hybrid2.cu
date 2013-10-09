
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

int
run (FMMWrapper_t *F)
{
  struct stopwatch_t* timer = NULL;
  long double t_total;
  timer = stopwatch_create ();

  stopwatch_start (timer);
  fprintf (stderr, "Performing Ulist on the CPU\n");
  omp_set_nested (1);
#pragma omp parallel shared(F) num_threads(2)
  {
#pragma omp sections
    {
      /* GPU section */
#pragma omp section
      {
        up_calc__gpu (F);
        vlist_calc__gpu (F);
				// xlist_calc__gpu (F);
				// wlist_calc__gpu (F);
        down_calc__gpu (F);
        cudaThreadSynchronize ();
      }

      /* CPU section */
#pragma omp section
      {
        ulist_calc__cpu (F);
      }
    }
  }
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
