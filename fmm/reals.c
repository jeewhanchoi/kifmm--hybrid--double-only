/**
 *  \file reals.c
 *
 *  \brief Implements some utility routines related to allocating
 *  vectors of real numbers.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "reals.h"

real_t *
reals_alloc (size_t n)
{
  real_t* x = NULL;
  if (n) {
    x = (real_t *)malloc (n * sizeof (real_t));
    assert (x);
  }
  return x;
}

void
reals_zero (size_t n, real_t* x)
{
  assert (x || !n);
  if (n)
    bzero (x, n * sizeof (real_t));
}

void
reals_copy (size_t n, const real_t* restrict src, real_t* restrict dest)
{
  if (n) {
    assert (src);
    assert (dest);
    memcpy (dest, src, n * sizeof (real_t));
  }
}

void
reals_free (real_t* x)
{
  if (x) free (x);
}

/* eof */
