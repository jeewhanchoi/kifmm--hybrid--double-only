#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timing.h"

/**
 *  Analyzes raw execution times, prints a summary to stdout, and
 *  returns the minimum execution time.
 */
long double
fprint_perf_summary (const char* tag, long double flops, long double bytes,
		     const long double* times, size_t trials,
		     FILE* fp)
{
  long double t_min = 0.0, t_mean = 0.0, t_std = 0.0, t_max = 0.0;

  /* Preconditions */
  assert (tag);
  assert (times || !trials);
  assert (fp);

  fprintf (fp, "=== Performance summary: %s ===\n", tag);

  /* Compute statistics */
  if (trials) {
    size_t k;
    long double t_sum = times[0];

    /* Compute min/max/mean */
    t_min = t_max = times[0];
    for (k = 1; k < trials; ++k) {
      double t = times[k];
      if (t < t_min) t_min = t;
      if (t > t_max) t_max = t;
      t_sum += t;
    }
    t_mean = t_sum / trials;

    /* Compute std dev */
    if (trials < 2)
      t_std = 0.0;
    else {
      t_sum = 0;
      for (k = 0; k < trials; ++k) {
	double t = times[k] - t_mean;
	t_sum += t * t;
      }
      t_std = sqrt (t_sum / (trials-1));
    }
  }

  flops *= 1e-9;
  bytes *= 1e-9;
  fprintf (fp, "  %Lg Gflops, %Lg GB\n", flops, bytes);
  fprintf (fp, "  Min: %Lg s -- %.3Lg Gflop/s, %.3Lg GB/s\n", t_min, flops / t_min, bytes / t_min);
  fprintf (fp, "  Mean: %Lg s -- %.3Lg Gflop/s, %.3Lg GB/s\n", t_mean, flops / t_mean, bytes / t_min);
  fprintf (fp, "  Max: %Lg s -- %.3Lg Gflop/s, %.3Lg GB/s\n", t_max, flops / t_max, bytes / t_min);
  fprintf (fp, "  Stddev: %Lg s (+/- %.4Lg%%)\n", t_std, 1e2 * t_std / t_mean);

  return t_min;
}

/****************************************************************
 * PAPI cycle counter
 ****************************************************************/

#if !defined(HAVE_TIMER) && defined(USE_PAPI)
#  define TIMER_DESC "PAPI"
#  include <papi.h>

#  define USE_STD_CREATE
#  define USE_STD_DESTROY

#define EVENT_ENV_VAR "EVENT"
#define NUM_TRIALS 5

static int event_set_ = PAPI_NULL;

static
void
assert_papi_ (int code, const char* file, size_t line)
{
  if (code != PAPI_OK) {
    const char* msg = PAPI_strerror (code);
    fprintf (stderr, "*** [%s:%lu] PAPI Error %d: %s ***\n",
	     file ? file : "(unknown file)", (unsigned long)line,
	     code, msg);
    assert (code == PAPI_OK);
    exit (1);
  }
}

#define assert_papi(c)  assert_papi_((c), __FILE__, __LINE__);

/* Determines the desired event to measure */
static
int
get_event_code_from_env (const char* env_var, char event_name[PAPI_MAX_STR_LEN])
{
  int event_code;
  int retval;

  char event_name_buffer[PAPI_MAX_STR_LEN];
  const char* event_name_env_str = getenv (env_var ? env_var : EVENT_ENV_VAR);

  if (!event_name_env_str)
    event_name_env_str = "PAPI_TOT_CYC";
  assert (event_name_env_str);

  memset (event_name_buffer, 0, sizeof (char) * PAPI_MAX_STR_LEN);
  strncpy (event_name_buffer, event_name_env_str, PAPI_MAX_STR_LEN-1);
  retval = PAPI_event_name_to_code (event_name_buffer, &event_code);
  if (retval != PAPI_OK) {
    /* Try to interpret as a hex code */
    sscanf (event_name_buffer, "%x", &event_code);
    if (event_code) {
      PAPI_event_code_to_name (event_code, event_name_buffer);
    } else {
      fprintf (stderr,
	       "*** Error: The desired event name, '%s', is not available on this system. ***\n",
	       event_name_buffer);
      assert (retval == PAPI_OK);
      exit (1);
    }
  }

  if (event_name)
    strcpy (event_name, event_name_buffer);
  return event_code;
}

static
void
init_papi (int* p_event_code, char p_event_name[PAPI_MAX_STR_LEN])
{
  int retval;
  int event_code;

  /* Initialize PAPI library */
  retval = PAPI_library_init (PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT) {
    fprintf (stderr, "*** Error: Couldn't initialize PAPI library. ***\n");
    assert (retval == PAPI_VER_CURRENT);
    exit (1);
  }

  event_code = get_event_code_from_env (NULL, p_event_name);
  if (p_event_code) *p_event_code = event_code;

  if (event_set_ == PAPI_NULL) {
    assert_papi (PAPI_create_eventset (&event_set_));
    assert_papi (PAPI_add_event (event_set_, event_code));
  }
}

static
long double
timer_resolution (void)
{
  long double sum = 0.0;
  int i;
  for (i = 0; i < NUM_TRIALS; ++i) {
    long_long t0;
    long_long t1;
    assert_papi (PAPI_start (event_set_));
    assert_papi (PAPI_read (event_set_, &t0));
    assert_papi (PAPI_stop (event_set_, &t1));
    sum += (long double)t1 - t0;
  }
  return (sum <= 0) ? (long double)1.0 : (sum / NUM_TRIALS);
}

/* ======= */

struct stopwatch_t
{
  long_long t_start_;
  long_long t_stop_;
  int is_running_;
};

void
stopwatch_init (void)
{
  if (event_set_ == PAPI_NULL) {
    int retval;
    char event_name[PAPI_MAX_STR_LEN];
    int event_code;
    
    fprintf (stderr, "Timer: %s\n", TIMER_DESC);
    
    init_papi (&event_code, event_name);
    
    fprintf (stderr, "Event counter: %s (code=0x%x)\n", event_name, event_code);
    fprintf (stderr, "Timer resolution: %Lg units\n", timer_resolution ());
    fflush (stderr);
  }
}

void
stopwatch_start (struct stopwatch_t* T)
{
  assert (T);
  T->is_running_ = 1;
  assert_papi (PAPI_start (event_set_));
  assert_papi (PAPI_read (event_set_, &T->t_start_));
}

long double
stopwatch_stop (struct stopwatch_t* T)
{
  if (T && T->is_running_) {
    assert_papi (PAPI_stop (event_set_, &T->t_stop_));
    assert_papi (PAPI_reset (event_set_));
    T->is_running_ = 0;
  }
  return stopwatch_elapsed (T);
}

long double
stopwatch_elapsed (struct stopwatch_t* T)
{
  long double dt = 0;
  if (T) {
    if (T->is_running_) {
      long_long t_cur;
      assert_papi (PAPI_read (event_set_, &t_cur));
      dt = (long double)t_cur - T->t_start_;
    } else {
      dt = (long double)T->t_stop_ - T->t_start_;
    }
  }
  return dt;
}

#  define HAVE_TIMER 1
#endif /* !defined(HAVE_TIMER) && defined(USE_PAPI) */

/****************************************************************
 * clock_gettime: POSIX high-resolution timer
 ****************************************************************/

#include <time.h>

#if !defined(HAVE_TIMER) && (defined(CLOCK_HIGHRES) || defined(CLOCK_REALTIME))
#  define TIMER_DESC "clock_gettime: POSIX high-resolution timer"

#  if defined(CLOCK_HIGHRES)
#    define CLOCK CLOCK_HIGHRES
#  else /* defined(CLOCK_REALTIME) */
#    define CLOCK CLOCK_REALTIME
#  endif

#define USE_STD_CREATE
#define USE_STD_DESTROY

static
long double
timespec_to_ldbl (struct timespec x)
{
  return x.tv_sec + 1.0E-9 * x.tv_nsec;
}

static
long double
timespec_diff (struct timespec start, struct timespec finish)
{
  long double out;
  out = finish.tv_nsec - (double)start.tv_nsec;
  out *= 1.0E-9L;
  out += finish.tv_sec - (double)start.tv_sec;
  return out;
}

static
long double
timer_resolution (void)
{
  struct timespec x;
  clock_getres (CLOCK, &x);
  return timespec_to_ldbl (x);
}

static
void
get_time (struct timespec* x)
{
  clock_gettime (CLOCK, x);
}

/* ======= */

struct stopwatch_t
{
  struct timespec t_start_;
  struct timespec t_stop_;
  int is_running_;
};

void
stopwatch_init (void)
{
  fprintf (stderr, "Timer: %s\n", TIMER_DESC);
  fprintf (stderr, "Timer resolution: %Lg\n", timer_resolution ());
  fflush (stderr);
}

void
stopwatch_start (struct stopwatch_t* T)
{
  assert (T);
  T->is_running_ = 1;
  get_time (&(T->t_start_));
}

long double
stopwatch_stop (struct stopwatch_t* T)
{
  if (T && T->is_running_) {
    get_time (&(T->t_stop_));
    T->is_running_ = 0;
  }
  return stopwatch_elapsed (T);
}

long double
stopwatch_elapsed (struct stopwatch_t* T)
{
  long double dt = 0;
  if (T) {
    if (T->is_running_) {
      struct timespec t_cur;
      get_time (&t_cur);
      dt = timespec_diff (T->t_start_, t_cur);
    } else {
      dt = timespec_diff (T->t_start_, T->t_stop_);
    }
  }
  return dt;
}

#  define HAVE_TIMER 1
#endif

/****************************************************************
 * gettimeofday: Better than nothing, I suppose.
 ****************************************************************/
#if !defined(HAVE_TIMER)
#  define TIMER_DESC "gettimeofday"

#define USE_STD_CREATE
#define USE_STD_DESTROY

#include <sys/time.h>

struct stopwatch_t
{
  struct timeval t_start_;
  struct timeval t_stop_;
  int is_running_;
};

void
stopwatch_init (void)
{
  fprintf (stderr, "Timer: %s\n", TIMER_DESC);
  fprintf (stderr, "Timer resolution: ~ 1 us (?)\n");
  fflush (stderr);
}

void
stopwatch_start (struct stopwatch_t* T)
{
  assert (T);
  T->is_running_ = 1;
  gettimeofday (&(T->t_start_), 0);
}

long double
stopwatch_stop (struct stopwatch_t* T)
{
  long double dt = 0;
  if (T) {
    if (T->is_running_) {
      gettimeofday (&(T->t_stop_), 0);
      T->is_running_ = 0;
    }
    dt = stopwatch_elapsed (T);
  }
  return dt;
}

static
long double
elapsed (struct timeval start, struct timeval stop)
{
  return (long double)(stop.tv_sec - start.tv_sec)
    + (long double)(stop.tv_usec - start.tv_usec)*1e-6;
}

long double
stopwatch_elapsed (struct stopwatch_t* T)
{
  long double dt = 0;
  if (T) {
    if (T->is_running_) {
      struct timeval stop;
      gettimeofday (&stop, 0);
      dt = elapsed (T->t_start_, stop);
    } else {
      dt = elapsed (T->t_start_, T->t_stop_);
    }
  }
  return dt;
}

#  define HAVE_TIMER 1
#endif

/****************************************************************
 * Base-case: No portable timer found.
 ****************************************************************/
#if !defined(HAVE_TIMER)
#  error "Can't find a suitable timer for this platform! Edit 'timer.c' to define one."
#endif

/****************************************************************
 * "Generic" methods that many timers can re-use.
 * (A hack to emulates C++-style inheritance.)
 ****************************************************************/

#if defined(USE_STD_CREATE)
struct stopwatch_t *
stopwatch_create (void)
{
  struct stopwatch_t* new_timer =
    (struct stopwatch_t *)malloc (sizeof (struct stopwatch_t));
  if (new_timer)
    memset (new_timer, 0, sizeof (struct stopwatch_t));
  return new_timer;
}
#endif

#if defined(USE_STD_DESTROY)
void
stopwatch_destroy (struct stopwatch_t* T)
{
  if (T) {
    stopwatch_stop (T);
    free (T);
  }
}
#endif

/* eof */

