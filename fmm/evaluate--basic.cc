#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "node.h"
#include "evaluate.h"
#include "evaluate--basic.h"
#include "reals_aligned.h"
#include "util.h"
#include "timing.h"


/* ------------------------------------------------------------------------
 */
FMMWrapper_t *
preproc (AllNodes *All_N)
{
  FMMWrapper_t* W = (FMMWrapper_t *) malloc(sizeof (FMMWrapper_t));
  assert (W);
  vector<NodeTree>& nodeVec = *All_N->N;

  W->AN = All_N; 
  return W;
}
/* ------------------------------------------------------------------------
 */
int
run (FMMWrapper_t *F)
{
  struct stopwatch_t* timer = NULL;
  long double t_up, t_u, t_v, t_w, t_x, t_down, t_total;
  timer = stopwatch_create ();

  stopwatch_start (timer);
  fprintf (stderr, "Performing Up calculation ...\n"); 
  stopwatch_start (timer);
  up_calc__cpu (F);
  t_up = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Up.\n");

  fprintf (stderr, "Performing U-list calculation (direct evaluation)...\n");
  ulist_calc__cpu (F);
  t_u = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Ulist.\n");

  fprintf (stderr, "Performing V-list calculation (pointwise multiply)...\n");
  vlist_calc__cpu (F);
  t_v = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Vlist.\n ");

  fprintf (stderr, "Performing W-list calculation...\n");
	wlist_calc__cpu (F);
  t_w = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Wlist.\n ");
  
  fprintf (stderr, "Performing X-list calculation...\n");
	xlist_calc__cpu (F);
  t_x = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Xlist.\n ");

  fprintf (stderr, "Performing Down calculation ...\n");
  down_calc__cpu (F);
  t_down = stopwatch_elapsed (timer);
  fprintf (stderr, "Done Down.\n");

  copy_trg_val__cpu (F);
  t_total = stopwatch_stop (timer);
  
  fprintf(stderr, "=== Statistics ===\n");
  fprintf (stdout, "Code: %s\n", get_implementation_name ());
  fprintf (stdout, "Floating-point word size: %lu bytes\n", sizeof (real_t));
  fprintf (stderr, "== Timing for FMM3d ==\n");
  fprintf (stderr, "  Up    : %Lg secs (%.1Lf%%)\n", t_up, t_up / t_total * 100);
  fprintf (stderr, "  U list: %Lg secs (%.1Lf%%)\n", (t_u - t_up), (t_u - t_up) / t_total * 100);
  fprintf (stderr, "  V list: %Lg secs (%.1Lf%%)\n", (t_v - t_u), (t_v - t_u) / t_total * 100);
  fprintf (stderr, "  W list: %Lg secs (%.1Lf%%)\n", (t_w - t_v), (t_w - t_v) / t_total * 100);
  fprintf (stderr, "  X list: %Lg secs (%.1Lf%%)\n", (t_x - t_w), (t_x - t_w) / t_total * 100);
  fprintf (stderr, "  Down  : %Lg secs (%.1Lf%%)\n", (t_down - t_x), (t_down - t_x) / t_total * 100);
  fprintf (stderr, "  ==> Total Execution Time: %Lg secs\n", t_total);

  stopwatch_destroy (timer);

  return 0;
}
/* ----------------------------------------------------------------------------------------------------------
 * eof
 */
