/**
 *  \file reals.h
 *
 *  \brief Implements some utility routines related to allocating
 *  vectors of real numbers.
 */

#if !defined (INC_REALS_H)
#define INC_REALS_H /*!< reals.h included. */

#if !defined (USE_FLOAT)
/** \brief Floating-point type for a real number. */
typedef double real_t;
#else
/** \brief Floating-point type for a real number. */
typedef float real_t;
#endif

#include <stddef.h>

#if defined (__cplusplus)
extern "C" {
#endif

  /** 4-D point */
  typedef struct {
    real_t x;
    real_t y;
    real_t z;
    real_t w;
  } point4_t;

  /** Returns a newly allocated array of 'n' real numbers. */
  real_t* reals_alloc (size_t n);

  /** Initializes 'n' array elements to zero. */
  void reals_zero (size_t n, real_t* x);

  /** Copies elements from one real array to another. */
  void reals_copy (size_t n, const real_t* restrict src, real_t* restrict dest);

  /** Frees a previously allocated array of real numbers. */
  void reals_free (real_t* x);

#if defined (__cplusplus)
}
#endif

#endif /* !defined (INC_REALS_H) */

/* eof */
