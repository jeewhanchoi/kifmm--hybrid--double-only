# KIFMM benchmark 

This repository contains CPU+GPU benchmark implementations of the
kernel-independent fast multipole method (KIFMM), which was used most
recently in the following papers:

- Jee Choi, Aparna Chandramowlishwaran, Kamesh Madduri, and Richard
  Vuduc. "A CPU-GPU hybrid implementation and model-driven scheduling
  of the fast multipole method." In _Proceedings of the 7th Workshop
  on General-Purpose Processing using GPUs (GPGPU-7)_, Salt Lake City,
  UT, USA,
  March 2014. [doi:10.1145/2576779.2576787](http://dl.acm.org/citation.cfm?id=2576787)
 
- Aparna Chandramowlishwaran, Jee Choi, Kamesh Madduri, and Richard
  Vuduc. "Brief announcement: Towards a communication optimal fast
  multipole method and its implications at exascale." In _Proceedings
  of the 24th Annual ACM Symposium on Parallel Algorithms and
  Architectures (SPAA'12)_, Pittsburgh, PA, USA, June
  25-27, 2012. [doi:10.1145/2312005.2312039](http://dl.acm.org/citation.cfm?id=2312039).

Feel free to post issues or pull requests on GitHub:
https://github.com/jeewhanchoi/kifmm--hybrid--double-only


## Compiling

From the root directory, run:

```bash
    make clean; make
```

The build will generate several instances of the benchmark as separate
executables (see **Description** below).

## Executing

```bash
    fmmd--{naive,omp,omp_sse,omp_sse_block,cuda,hybrid1,hybrid2,hybrid3} \
	    <# pts> {uniform,ellipseUniformAngles} <# pts-per-box>
```


## Description

This benchmark includes CPU, GPU, and CPU+GPU hybrid implementations
for the fast multipole method (FMM) in double-precision.

The benchmark generates the follwing executables,

- `fmmd--naive`: Baseline sequential code
- `fmmd--omp`: OpenMP parallelized code
- `fmmd--omp_sse`: OpenMP parallelized + SIMD vectorized code
- `fmmd--omp_sse_block`: OpenMP parallelized + SIMD vectorized +
  Blocking (translation vector) + Blocking (up)
- `fmmd--cuda`: CUDA code
- `fmmd--hybrid1`: U-list on GPU; up, V-list, down on CPU
- `fmmd--hybrid2`: U-list on CPU; up, V-list, down on GPU
- `fmmd--hybrid3`: A optimal schedule for non-uniform distributions on
  hybrid CPU-GPU systems

## Environment Variables

- NUMA-aware memory allocation can be set/unset using the environment
  variable, `NUMA`. Default = `yes`.

> Note: When doing NUMA-aware memory allocation, threads must be
> pinned appropriately by, for instance, using the appropriate
> environment variable, e.g., `KMP_AFFINITY` (icc) or `GOMP_ AFFINITY`
> (gcc). Example:
>
> ```bash
>     export KMP_AFFINITY=granularity=fine,compact,1,0,verbose # Without hyperthreading; or:
>     export KMP_AFFINITY=granularity=fine,compact,verbose # with hyperthreading
> ```

- Number of threads can be varied by changing the environment
  variable, `OMP_NUM_THREADS`. Default = Max # of threads.

- When blocking is enabled, translation block size can be set by the
  environment variable, `BS`. Default = 49.

- When blocking is enabled, up block size can be set by the
  environment variable, `UPBS`. Default = 1000.
 
- Accuracy can be varied by changing the environment variable,
  `NP`. Default = 6.

> On the GPU, only 3 precision are supported (`NP=3`, `NP=4`, and
> `NP=6`). You must also set `#define NP_(X)` in `cuda.cu` to 1 (all
> others should be set to 0) and re-compiled in order for the GPU
> version to work. Lastly, `env NP=<X>` must also be set as the GPU
> code uses CPU's tree construction code.

- The error is computed by taking a random sample. This paramter can
  be varied by changing the environment variable, `NV`. Default =
  1000.


## Example usage

```bash
	# with NP_6 set to 1, GPU FMM with uniform 
    # distribution and 6 digits of precision:
    ./fmmd--cuda 4194304 uniform 512
	
	# with NP_4 set to 1, GPU FMM with uniform
	# distribution and 4 digits of precision:
    env NP=4 ./fmmd--cuda 1048576 uniform 512
	
	# Most optimized CPU FMM with 3 digits of
	# precision:
    env NP=3 ./fmmd--omp_sse_block 1048576 uniform 128
```
	
## Other notes

- By default, for the GPU version, all data structures are allocated
in the memory prior to execution.  However, due to CUFFT taking too
much memory, for large number of points and/or higher precision, there
is support to allocate data as they are needed.  This can be
automatically turned on and off setting the `MIN_DATA` definition
found in `partial.h` to `1` or `0`.


Contributors
------------

- Jee Whan Choi <jee@gatech.edu>
- Aparna Chandramowlishwaran <amowli@uci.edu>
- Kamesh Madduri <madduri@cse.psu.edu>
- Richard Vuduc <richie@cc.gatech.edu>
