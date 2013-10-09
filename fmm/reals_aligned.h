/**
 *  \file reals_aligned.h
 *
 *  \brief Implements some utility routines related to allocating
 *  aligned vectors of real numbers.
 */

#if !defined (INC_REALS_ALIGNED_H)
#define INC_REALS_ALIGNED_H /*!< reals_aligned.h included. */

#include "reals.h"

#if !defined (IDEAL_ALIGNMENT)
/** Default alignment, in bytes. */
#  define IDEAL_ALIGNMENT 16
#endif

#if !defined (BEST_ALIGNMENT_ATTR)
#  define BEST_ALIGNMENT_ATTR __attribute__ ((aligned (64)))
#else
#  define BEST_ALIGNMENT_ATTR
#endif

/** \brief No. of real_t words / SIMD register. */
#define SIMD_LEN (IDEAL_ALIGNMENT / sizeof (real_t))

/** \brief Qualifier to declare a static array aligned. */
#define DECL_SIMD_ALIGNED  __declspec(align(IDEAL_ALIGNMENT))

#if !defined (USE_FLOAT)
#  if !defined (__SSE2__)
#    error "*** SSE2 is not available. ***"
#  endif
#if !defined (USE_NVCC)
#  include <emmintrin.h>
/** \name SIMD instructions */
/*@{*/
#  define SIMD_REG      __m128d
#  define SIMD_LOAD     _mm_load_pd
#  define SIMD_STORE    _mm_store_pd
#  define SIMD_LOAD_U   _mm_loadu_pd
#  define SIMD_STORE_U  _mm_storeu_pd
#  define SIMD_STORE_S  _mm_store_sd
#  define SIMD_LOAD1    _mm_load1_pd
#  define SIMD_SET      _mm_set_pd
#  define SIMD_SET1     _mm_set1_pd
#  define SIMD_ZERO     SIMD_SET1 (0.0)
#  define SIMD_SUB      _mm_sub_pd
#  define SIMD_ADD      _mm_add_pd
#  define SIMD_MUL      _mm_mul_pd
#  define SIMD_DIV      _mm_div_pd
#  define SIMD_XOR      _mm_xor_pd
#  define SIMD_ANDNOT   _mm_andnot_pd
#  define SIMD_SHUFFLE  _mm_shuffle_pd
#  define SIMD_SQRT     _mm_sqrt_pd
#  define SIMD_INV(x)   SIMD_DIV (SIMD_SET1 (1.0), (x))
#  define SIMD_CMPEQ    _mm_cmpeq_pd
#  define SIMD_CVTD_S   _mm_cvtpd_ps
#  define SIMD_CVTS_D   _mm_cvtps_pd
#  define SIMD_RSQRT_S  _mm_rsqrt_ps
#  define SIMD_INV_SQRT_S(x) SIMD_CVTS_D (SIMD_RSQRT_S (SIMD_CVTD_S(x)))
#  define SIMD_INV_SQRT(x)   SIMD_INV(SIMD_SQRT(x))

#  define SIMD_SET_1    SIMD_SET (1.0, -1.0)
#  define SHUFFLE_0  _MM_SHUFFLE2 (0, 0)
#  define SHUFFLE_1  _MM_SHUFFLE2 (1, 1)
#  define SHUFFLE_2  _MM_SHUFFLE2 (0, 1)
/*@}*/
#endif
/** FFT functions */
#  define FFT_COMPLEX   fftw_complex
#  define FFT_PLAN    fftw_plan

#  define FFT_CREATE  fftw_plan_many_dft_r2c
#  define FFT_RE_EXECUTE fftw_execute_dft_r2c
#  define FFT_EXECUTE fftw_execute

#  define IFFT_CREATE  fftw_plan_many_dft_c2r
#  define IFFT_EXECUTE fftw_execute
#  define IFFT_RE_EXECUTE fftw_execute_dft_c2r
#  define FFT_DESTROY  fftw_destroy_plan 

/** BLAS functions */
#  define _GESVD  DGESVD
#  define _AXPY  DAXPY
#  define _GEMV  DGEMV
#  define _GEMM  DGEMM

#else
#if !defined (USE_NVCC)
#  include <xmmintrin.h>
/** \name SIMD instructions */
/*@{*/
#  define SIMD_REG      __m128
#  define SIMD_LOAD     _mm_load_ps
#  define SIMD_STORE    _mm_store_ps
#  define SIMD_STORE_S  _mm_store_ss
#  define SIMD_LOAD_U   _mm_loadu_ps
#  define SIMD_STORE_U  _mm_storeu_ps
#  define SIMD_LOAD1    _mm_load1_ps
#  define SIMD_SET      _mm_set_ps
#  define SIMD_SET1     _mm_set1_ps
#  define SIMD_ZERO     SIMD_SET1 (0.0)
#  define SIMD_SUB      _mm_sub_ps
#  define SIMD_ADD      _mm_add_ps
#  define SIMD_MUL      _mm_mul_ps
#  define SIMD_DIV      _mm_div_ps
#  define SIMD_SQRT     _mm_sqrt_ps
#  define SIMD_XOR      _mm_xor_ps
#  define SIMD_ANDNOT   _mm_andnot_ps
#  define SIMD_SHUFFLE  _mm_shuffle_ps
#  define SIMD_INV_SQRT _mm_rsqrt_ps
#  define SIMD_INV      _mm_rcp_ps
#  define SIMD_CMPEQ    _mm_cmpeq_ps

#  define SIMD_SET_1    SIMD_SET (1.0, -1.0, 1.0, -1.0)
#  define SHUFFLE_0  _MM_SHUFFLE (2, 2, 0, 0)
#  define SHUFFLE_1  _MM_SHUFFLE (3, 3, 1, 1)
#  define SHUFFLE_2  _MM_SHUFFLE (2, 3, 0, 1)
/*@}*/
#endif
/** FFT functions */
#  define FFT_COMPLEX   fftwf_complex
#  define FFT_PLAN    fftwf_plan

#  define FFT_CREATE  fftwf_plan_many_dft_r2c
#  define FFT_RE_EXECUTE  fftwf_execute_dft_r2c
#  define FFT_EXECUTE fftwf_execute

#  define IFFT_CREATE  fftwf_plan_many_dft_c2r
#  define IFFT_EXECUTE fftwf_execute
#  define IFFT_RE_EXECUTE fftwf_execute_dft_c2r
#  define FFT_DESTROY  fftwf_destroy_plan 

/** BLAS functions */
#  define _GESVD  SGESVD
#  define _AXPY  SAXPY
#  define _GEMV  SGEMV
#  define _GEMM  SGEMM
#endif

#define PREFETCH_NT(address)  _mm_prefetch((address), _MM_HINT_NTA)

/*#include <pmmintrin.h>*/

#include <stddef.h>

#if defined (__cplusplus)
extern "C" {
#endif

  /**
   *  \brief Returns a newly allocated array of 'n' real numbers,
   *  aligned on some appropriate boundary for vectorization.
   */
  real_t* reals_alloc__aligned (size_t n);

  /** Initializes 'n' array elements to zero. */
  void reals_zero__aligned (size_t n, real_t* x);

  /**
   *  \brief Copies elements from one real array to another.
   *  \note Both 'src' and 'dest' arrays MUST be aligned.
   */
  void reals_copy__aligned (size_t n, const real_t* restrict src, real_t* restrict dest);

  /**
   *  \brief Copies elements from one real array to another.
   *  \note Only 'src' is aligned.
   */
  void reals_copy__alignedS (size_t n, const real_t* restrict src, real_t* restrict dest);

  /**
   *  \brief Copies elements from one real array to another.
   *  \note Only 'dest' is aligned.
   */
  void reals_copy__alignedD (size_t n, const real_t* restrict src, real_t* restrict dest);

  /**
   *  \brief Frees a previously allocated aligned array of real numbers.
   */
  void reals_free__aligned (real_t* x);

  /**
   *  \name 4-D point routines, analogous to scalar versions above.
   */
  /*@{*/
  point4_t* point4_alloc__aligned (size_t n);
  void point4_zero__aligned (size_t n, point4_t* P);
  void point4_free__aligned (point4_t* P);
  /*@}*/

#if defined (__cplusplus)
}
#endif

#endif /* !defined (INC_REALS_ALIGNED_H) */

/* eof */
