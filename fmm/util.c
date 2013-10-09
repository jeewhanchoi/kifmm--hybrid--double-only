#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

/* ------------------------------------------------------------------------
 */

int
getenv__int (const char* var, int default_value)
{
  int value = default_value;
  int temp;
  const char* str_value;
  if (var && (str_value = getenv (var)) && sscanf (str_value, "%d", &temp))
    value = temp;
  return value;
}

/* ------------------------------------------------------------------------
 */

int
getenv__flag (const char* var, int default_value)
{
  int value = default_value;
  const char* str_value;
  if (var && (str_value = getenv (var))) {
    if (strcasecmp (str_value, "y") == 0
	|| strcasecmp (str_value, "yes") == 0
	|| strcasecmp (str_value, "t") == 0
	|| strcasecmp (str_value, "true") == 0)
      value = 1;
    else if (strcasecmp (str_value, "n") == 0
	     || strcasecmp (str_value, "no") == 0
	     || strcasecmp (str_value, "f") == 0
	     || strcasecmp (str_value, "false") == 0)
      value = 0;
    else if (!sscanf (str_value, "%d", &value))
      value = default_value;
  }
  return value;
}

/* ------------------------------------------------------------------------
 */

/** Non-zero if NUMA desired */
static int numa_ = -1;

int
getenv__numa (void)
{
  if (numa_ < 0) { 
    const char* s = getenv ("NUMA");
    numa_ = 1; 
    if (s) {
      if (strcasecmp (s, "yes") == 0)
        numa_ = 1; 
      else {
        int v = atol (s); 
        if (v >= 0)
          numa_ = v; 
        else {
          fprintf (stderr,
            "*** WARNING: NUMA (='%s') is invalid. ***\n"
            "    Should be 'yes' or some value >= 0.\n"
            "    Using default value: %d\n",
            s, numa_);
        }
      }    
    }    
  }
  return numa_;
}

/* ------------------------------------------------------------------------
 */
/** Desired accuracy. Must be > 0. */
static int np_ = 0;

/**
 *  \brief If global 'np_' is not yet set to a valid value,
 *  check if user requested a value via the environment variable,
 *  NP, and otherwise return some default value.
 */
int
getenv__accuracy (void)
{
  if (!np_) {
    np_ = 6; /* default value */
    const char* s = getenv ("NP");
    if (s) {
      long t = atof (s); 
      if (t <= 0) { 
        fprintf (stderr, "*** WARNING: NP (=%ld) should be >= 1 ***\n", t);
        fprintf (stderr, "    Using default value: %lu\n", (unsigned long)np_);
      }    
      else /* t >= 1 */
        np_ = (size_t)t;
    }    
  }
  return np_;
}

/* ------------------------------------------------------------------------
 */
/** Block size. Must be > 0. */
static int bs_ = 0;

/**
 *  \brief If global 'bs_' is not yet set to a valid value,
 *  check if user requested a value via the environment variable,
 *  BS, and otherwise return some default value.
 */
int
getenv__block_size (void)
{
  if (!bs_) {
    bs_ = 49; /* default value */
    const char* s = getenv ("BS");
    if (s) {
      long t = atof (s); 
      if (t <= 0) { 
        fprintf (stderr, "*** WARNING: BS (=%ld) should be >= 1 ***\n", t);
        fprintf (stderr, "    Using default value: %lu\n", (unsigned long)bs_);
      }    
      else /* t >= 1 */
        bs_ = (size_t)t;
    }    
  }
  return bs_;
}

/* ------------------------------------------------------------------------
 */
/** Block size. Must be > 0. */
static int upbs_ = 0;

/**
 *  \brief If global 'upbs_' is not yet set to a valid value,
 *  check if user requested a value via the environment variable,
 *  UPBS, and otherwise return some default value.
 */
int
getenv__upbs (void)
{
  if (!upbs_) {
    upbs_ = 1000; /* default value */
    const char* s = getenv ("UPBS");
    if (s) {
      long t = atof (s); 
      if (t <= 0) { 
        fprintf (stderr, "*** WARNING: UPBS (=%ld) should be >= 1 ***\n", t);
        fprintf (stderr, "    Using default value: %lu\n", (unsigned long)upbs_);
      }    
      else /* t >= 1 */
        upbs_ = (size_t)t;
    }    
  }
  return upbs_;
}

/* ------------------------------------------------------------------------
 */
/** Number of points to check during validation. Must be >= 0. */
static int num_chk_ = 0;

/**
 *  \brief If global 'num_chk_' is not yet set to a valid value,
 *  check if user requested a value via the environment variable,
 *  NV, and otherwise return some default value.
 */
int
getenv__validate (void)
{
  if (!num_chk_) {
    num_chk_ = 1000; /* default value */
    const char* s = getenv ("NV");
    if (s) {
      long t = atof (s); 
      if (t <= 0) { 
        fprintf (stderr, "*** WARNING: NV (=%ld) should be >= 0 ***\n", t);
        fprintf (stderr, "    Using default value: %lu\n", (unsigned long)num_chk_);
      }    
      else /* t >= 0 */
        num_chk_ = (size_t)t;
    }    
  }
  return num_chk_;
}

/* ------------------------------------------------------------------------
 */
/** Sources and targets coincide?. Must be 0 or 1. */
static int c_ = 0;

/**
 *  \brief If global 'c_' is not yet set to a valid value,
 *  check if user requested a value via the environment variable,
 *  C, and otherwise return some default value.
 */
int
getenv__coincide (void)
{
  if (!c_) {
    c_ = 0; /* default value */
    const char* s = getenv ("C");
    if (s) {
      long t = atof (s); 
      if (t < 0 || t > 1 ) { 
        fprintf (stderr, "*** WARNING: C (=%ld) should be 0 or 1 ***\n", t);
        fprintf (stderr, "    Using default value: %lu\n", (unsigned long)c_);
      }    
      else 
        c_ = (size_t)t;
    }    
  }
  return c_;
}


/* eof */
