#if !defined(TIMING_H_)
#define TIMING_H_ 1

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct stopwatch_t;

/** \brief Initialize timing library. Call ONCE per application run. */
void stopwatch_init (void);

/** \brief Creates a 'stopwatch'. */
struct stopwatch_t* stopwatch_create (void);

/** \brief Destroy a 'stopwatch'. */
void stopwatch_destroy (struct stopwatch_t *);

/** \brief Turn stopwatch on. */
void stopwatch_start (struct stopwatch_t *);

/** \brief Turn stopwatch off and return elapsed time, in seconds. */
long double stopwatch_stop (struct stopwatch_t *);

/** \brief If on, return elapsed time since last call to stopwatch_start();
 * otherwise, return elapsed time at last call to stopwatch_stop().
 */
long double stopwatch_elapsed (struct stopwatch_t *);

#define TIMING_MAX_INNER_ITERATIONS 1000000000

/** Auto-calibrating timing loop */
#define TIMING_LOOP(time_trials, num_trials, verbose, PRE, CODE, POST) \
  { \
    size_t num_inner__; \
    size_t t__; \
    struct stopwatch_t* timer__ = stopwatch_create (); \
    const char* env_inner = getenv ("NUM_INNER"); \
    assert (timer__); \
    assert (time_trials || !num_trials); \
    if (verbose) { \
      fprintf (stderr, "Determining no. of inner iterations..."); \
      fflush (stderr); \
    } \
    /* Auto-detect/override no. of inner iterations */ \
    if (env_inner && atol (env_inner) > 0) \
      num_inner__ = (size_t)atol (env_inner); \
    else { \
      num_inner__ = 0; \
      do { \
        size_t i__; \
        PRE; \
        if (!num_inner__) \
          num_inner__ = 1; \
        else if (num_inner__ < 4) \
          ++num_inner__; \
        else \
          num_inner__ = num_inner__ * 5 / 4; \
        stopwatch_start (timer__); \
        for (i__ = 0; i__ < num_inner__; ++i__) \
        { \
	  CODE; \
        } \
        stopwatch_stop (timer__); \
        POST; \
        /* if (verbose) { fprintf (stderr, "%lu (%Lf s)...", (unsigned long)num_inner__, stopwatch_elapsed (timer__)); fflush (stderr); } */ \
      } while (stopwatch_elapsed (timer__) < .1 && num_inner__ < TIMING_MAX_INNER_ITERATIONS); \
    } \
    if (verbose) { \
      fprintf (stderr, "%lu\n", num_inner__); \
      fflush (stderr); \
    } \
    /* Outer timing loop */ \
    for (t__ = 0; t__ < num_trials; ++t__) { \
      size_t i__; \
      if (verbose) { \
	fprintf (stderr, "Time trial %lu: ", t__);	\
	fflush (stderr); \
      } \
      { \
	PRE; \
      } \
      stopwatch_start (timer__); \
      for (i__ = 0; i__ < num_inner__; ++i__) { \
	CODE; \
      } \
      stopwatch_stop (timer__); \
      { \
	POST; \
      } \
      time_trials[t__] = stopwatch_elapsed (timer__) / num_inner__; \
      if (verbose) { \
	fprintf (stderr, "%Lg s\n", time_trials[t__]); \
	fflush (stderr); \
      } \
    } \
    stopwatch_destroy (timer__); \
  }

/** \brief Prints a performance summary from raw times gathered using, say, the \ref TIMING_LOOP macro. */
long double fprint_perf_summary (const char* tag,
				 long double flops,
				 long double bytes,
				 const long double* times, size_t trials,
				 FILE* fp);

#ifdef __cplusplus
}
#endif

#endif /* TIMING_H_ */
