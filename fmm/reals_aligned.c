/**
 *  \file reals_aligned.c
 *
 *  \brief Implements some utility routines related to allocating
 *  vectors of real numbers.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <pmmintrin.h>
#include <mm_malloc.h>
#include "reals_aligned.h"

real_t *
reals_alloc__aligned (size_t n)
{
  real_t* x = NULL;
  if (n) {
    x = (real_t *)_mm_malloc (n * sizeof (real_t), IDEAL_ALIGNMENT);
    assert (x);
  }
  return x;
}

void
reals_zero__aligned (size_t n, real_t* x)
{
  assert (x || !n);
  if (n) {
    size_t i;
    for (i = 0; (i+SIMD_LEN) <= n; i += SIMD_LEN)
      SIMD_STORE (x+i, SIMD_ZERO);
    if (i < n)
      bzero (x+i, (n-i)*sizeof (real_t));
  }
}

void
reals_copy__aligned (size_t n, const real_t* restrict src, real_t* restrict dest)
{
  if (n) {
    size_t i;
    assert (src);
    assert (dest);
    for (i = 0; (i+SIMD_LEN) <= n; i += SIMD_LEN)
      SIMD_STORE (dest+i, SIMD_LOAD (src+i));
    if (i < n)
      memcpy (dest+i, src+i, (n-i) * sizeof (real_t));
  }
}

void
reals_copy__alignedS (size_t n, const real_t* restrict src, real_t* restrict dest)
{
  if (n) {
    size_t i;
    assert (src);
    assert (dest);
    for (i = 0; (i+SIMD_LEN) <= n; i += SIMD_LEN)
      SIMD_STORE_U (dest+i, SIMD_LOAD (src+i));
    if (i < n)
      memcpy (dest+i, src+i, (n-i) * sizeof (real_t));
  }
}

void
reals_copy__alignedD (size_t n, const real_t* restrict src, real_t* restrict dest)
{
  if (n) {
    size_t i;
    assert (src);
    assert (dest);
    for (i = 0; (i+SIMD_LEN) <= n; i += SIMD_LEN)
      SIMD_STORE (dest+i, SIMD_LOAD_U (src+i));
    if (i < n)
      memcpy (dest+i, src+i, (n-i) * sizeof (real_t));
  }
}

void
reals_free__aligned (real_t* x)
{
  if (x)
    _mm_free (x);
}

point4_t *
point4_alloc__aligned (size_t n)
{
  point4_t* x = NULL;
  if (n) {
    x = (point4_t *)_mm_malloc (n * sizeof (point4_t), IDEAL_ALIGNMENT);
    assert (x);
  }
  return x;
}

void
point4_zero__aligned (size_t n, point4_t* P)
{
  bzero (P, n * sizeof (point4_t));
}

void
point4_free__aligned (point4_t* P)
{
  if (P)
    _mm_free (P);
}

/* eof */
