top_srcdir = ..
include $(top_srcdir)/Makefile.inc

#=======================================================================

TARGETS =
TARGETS += fmmd--naive$(EXEEXT)

ifneq ($(OMPFLAGS),)
  TARGETS += fmmd--omp$(EXEEXT)
  TARGETS += fmmd--omp_sse$(EXEEXT)
  TARGETS += fmmd--omp_sse_block$(EXEEXT)
endif

ifneq ($(NVCC),)
  TARGETS += fmmd--cuda$(EXEEXT)
  TARGETS += fmmd--hybrid1$(EXEEXT)
  TARGETS += fmmd--hybrid2$(EXEEXT)
  TARGETS += fmmd--hybrid3$(EXEEXT)
endif

CLEANFILES = $(TARGETS)

all: $(TARGETS)

#=======================================================================
# directory for FFTW
FFTW3_DIR = /opt/fftw-3.2.2
FFTW3_INC = -I$(FFTW3_DIR)/include
FFTW3_LIBS = -L$(FFTW3_DIR)/lib -lfftw3

FFTW3F_DIR = /nethome/achandramo3/software/fftw3f
FFTW3F_INC = -I$(FFTW3F_DIR)/include
FFTW3F_LIBS = -L$(FFTW3F_DIR)/lib -lfftw3f

# directory for BLAS
BLAS_DIR = /opt/intel/mkl/10.2.2.025/lib/em64t
BLAS_LIBS = $(BLAS_DIR)/libmkl_solver_lp64.a  -Wl,--start-group $(BLAS_DIR)/libmkl_intel_lp64.a $(BLAS_DIR)/libmkl_intel_thread.a $(BLAS_DIR)/libmkl_core.a $(BLAS_DIR)/libiomp5.a -Wl,--end-group -lpthread 

# directory for CUDA and CUDA SDK Util
ifneq ($(NVCC),)
CUDA_INC = -I$(CUDA_DIR)/include -I$(CUDASDK_DIR)/common/inc 
CUDA_LIBS = -L$(CUDASDK_DIR)/lib -lcutil_x86_64 -L$(CUDA_DIR)/lib64 -lcudart -lcufft
endif

INCLUDES = $(FFTW3_INC)
INCLUDES += $(FFTW3F_INC)
INCLUDES += $(CUDA_INC)

LIBS = $(FFTW3_LIBS) $(FFTW3F_LIBS) $(BLAS_LIBS) $(CUDA_LIBS)

LDFLAGS = ${LIBS}

fmm_common_HDRS = reals.h reals_aligned.h util.h
fmm_common_SRCS = $(fmm_common_HDRS:.h=.c)
fmm_common_DEPS = $(fmm_common_SRCS:.c=.d)
fmm_common_LIBS = \
                  $(top_srcdir)/timing/libtiming.a 
-include $(fmm_common_DEPS)

CLEANFILES += $(fmm_common_DEPS)

#=======================================================================

fmmd_common_OBJS = $(fmm_common_SRCS:.c=.o__d) evaluate.o__d input.o__d driver.o__d svdrep.o__d vecmatop.o__d trans.o__d
fmmd_common_LIBS =

CLEANFILES += $(fmmd_common_OBJS)
#=======================================================================

fmm_naive_HDRS = evaluate.h evaluate--basic.h
fmm_naive_SRCS = evaluate--basic.cc naive.cc
fmm_naive_DEPS = $(fmm_basic_SRCS:.cc=.d)
-include $(fmm_naive_DEPS)

fmmd_naive_OBJS = $(fmm_naive_SRCS:.cc=.o__d)
fmmd_naive_LIBS =

fmmd--naive$(EXEEXT): \
	    $(fmmd_naive_OBJS) \
	    $(fmmd_naive_LIBS) \
	    $(fmmd_common_OBJS) \
	    $(fmmd_common_LIBS) \
	    $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) ${INCLUDES} -o $@ $^ $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS}

CLEANFILES += $(fmmd_naive_OBJS)
#=======================================================================

fmm_omp_HDRS = $(fmm_naive_HDRS)
fmm_omp_SRCS = evaluate--basic.cc omp.cc
fmm_omp_DEPS = $(fmm_omp_SRCS:.cc=.d)
-include $(fmm_omp_DEPS)

fmmd_omp_OBJS = $(fmm_omp_SRCS:.cc=.o__d)
fmmd_omp_LIBS =

fmmd--omp$(EXEEXT): \
	    $(fmmd_omp_OBJS) \
	    $(fmmd_omp_LIBS) \
	    $(fmmd_common_OBJS) \
	    $(fmmd_common_LIBS) \
	    $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) ${INCLUDES} -o $@ $^ $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS}

CLEANFILES += $(fmmd_omp_OBJS)
#=======================================================================

fmm_omp_sse_HDRS = $(fmm_naive_HDRS)
fmm_omp_sse_SRCS = evaluate--basic.cc omp_sse.cc
fmm_omp_sse_DEPS = $(fmm_omp_sse_SRCS:.cc=.d)
-include $(fmm_omp_sse_DEPS)

fmmd_omp_sse_OBJS = $(fmm_omp_sse_SRCS:.cc=.o__d)
fmmd_omp_sse_LIBS =

fmmd--omp_sse$(EXEEXT): \
	    $(fmmd_omp_sse_OBJS) \
	    $(fmmd_omp_sse_LIBS) \
	    $(fmmd_common_OBJS) \
	    $(fmmd_common_LIBS) \
	    $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) ${INCLUDES} -o $@ $^ $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS}

CLEANFILES += $(fmmd_omp_sse_OBJS)
#=======================================================================

fmm_omp_sse_block_HDRS = $(fmm_naive_HDRS)
fmm_omp_sse_block_SRCS = evaluate--basic.cc omp_sse_block.cc
fmm_omp_sse_block_DEPS = $(fmm_omp_sse_block_SRCS:.cc=.d)
-include $(fmm_omp_sse_block_DEPS)

fmmd_omp_sse_block_OBJS = $(fmm_omp_sse_block_SRCS:.cc=.o__d)
fmmd_omp_sse_block_LIBS =

fmmd--omp_sse_block$(EXEEXT): \
	    $(fmmd_omp_sse_block_OBJS) \
	    $(fmmd_omp_sse_block_LIBS) \
	    $(fmmd_common_OBJS) \
	    $(fmmd_common_LIBS) \
	    $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) ${INCLUDES} -o $@ $^ $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS}

CLEANFILES += $(fmmd_omp_sse_block_OBJS)
#=======================================================================

fmm_omp_sse_block_async_HDRS = $(fmm_naive_HDRS)
fmm_omp_sse_block_async_SRCS = evaluate--basic.c omp_sse_block_async.c
fmm_omp_sse_block_async_DEPS = $(fmm_omp_sse_block_async_SRCS:.c=.d)
-include $(fmm_omp_sse_block_async_DEPS)

fmmd_omp_sse_block_async_OBJS = $(fmm_omp_sse_block_async_SRCS:.c=.o__d)
fmmd_omp_sse_block_async_LIBS =

fmmd--omp_sse_block_async$(EXEEXT): \
	    $(fmmd_omp_sse_block_async_OBJS) \
	    $(fmmd_omp_sse_block_async_LIBS) \
	    $(fmmd_common_OBJS) \
	    $(fmmd_common_LIBS) \
	    $(fmm_common_LIBS)
	$(CC) $(CFLAGS_STD) $(CFLAGS) $(COPTFLAGS) -o $@ $^ $(OMPFLAGS) $(CLDFLAGS)

CLEANFILES += $(fmmd_omp_sse_block_async_OBJS)
#=======================================================================

fmm_cuda_HDRS = $(fmm_naive_HDRS) evaluate--cuda.h
fmm_cuda_SRCS =  
fmm_cuda_CUSRCS = evaluate--cuda.cu run--cuda.cu cuda.cu 
fmm_cuda_CXXSRCS = omp_sse_block.cc

fmmd_cuda_CUOBJS = $(fmm_cuda_CUSRCS:.cu=.o__d)
fmmd_cuda_CXXOBJS = $(fmm_cuda_CXXSRCS:.cc=.o__d) 
fmmd_cuda_LIBS =

fmmd--cuda$(EXEEXT): \
      $(fmmd_cuda_OBJS) \
      $(fmmd_cuda_CUOBJS) \
      $(fmmd_cuda_CXXOBJS) \
      $(fmmd_cuda_LIBS) \
      $(fmmd_common_OBJS) \
      $(fmmd_common_LIBS) \
      $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) -DUSE_NVCC ${INCLUDES} -o $@ $^ \
      $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS} 

CLEANFILES += $(fmmd_cuda_CUOBJS)
CLEANFILES += $(fmmd_cuda_CXXOBJS)
#=======================================================================

fmm_hybrid1_CUSRCS = evaluate--cuda.cu run--hybrid1.cu cuda.cu 
fmm_hybrid1_CXXSRCS = omp_sse_block.cc

fmmd_hybrid1_CUOBJS = $(fmm_hybrid1_CUSRCS:.cu=.o__d)
fmmd_hybrid1_CXXOBJS = $(fmm_hybrid1_CXXSRCS:.cc=.o__d) 
fmmd_hybrid1_LIBS =

fmmd--hybrid1$(EXEEXT): \
      $(fmmd_hybrid1_OBJS) \
      $(fmmd_hybrid1_CUOBJS) \
      $(fmmd_hybrid1_CXXOBJS) \
      $(fmmd_hybrid1_LIBS) \
      $(fmmd_common_OBJS) \
      $(fmmd_common_LIBS) \
      $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) -DUSE_NVCC ${INCLUDES} -o $@ $^ \
      $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS} 

CLEANFILES += $(fmmd_hybrid1_CUOBJS)
CLEANFILES += $(fmmd_hybrid1_CXXOBJS)
#=======================================================================
fmm_hybrid2_CUSRCS = evaluate--cuda.cu run--hybrid2.cu cuda.cu 
fmm_hybrid2_CXXSRCS = omp_sse_block.cc

fmmd_hybrid2_CUOBJS = $(fmm_hybrid2_CUSRCS:.cu=.o__d)
fmmd_hybrid2_CXXOBJS = $(fmm_hybrid2_CXXSRCS:.cc=.o__d) 
fmmd_hybrid2_LIBS =

fmmd--hybrid2$(EXEEXT): \
      $(fmmd_hybrid2_OBJS) \
      $(fmmd_hybrid2_CUOBJS) \
      $(fmmd_hybrid2_CXXOBJS) \
      $(fmmd_hybrid2_LIBS) \
      $(fmmd_common_OBJS) \
      $(fmmd_common_LIBS) \
      $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) -DUSE_NVCC ${INCLUDES} -o $@ $^ \
      $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS} 

CLEANFILES += $(fmmd_hybrid2_CUOBJS)
CLEANFILES += $(fmmd_hybrid2_CXXOBJS)

#=======================================================================
fmm_hybrid3_CUSRCS = evaluate--cuda.cu run--hybrid3.cu cuda.cu 
fmm_hybrid3_CXXSRCS = omp_sse_block.cc

fmmd_hybrid3_CUOBJS = $(fmm_hybrid3_CUSRCS:.cu=.o__d)
fmmd_hybrid3_CXXOBJS = $(fmm_hybrid3_CXXSRCS:.cc=.o__d) 
fmmd_hybrid3_LIBS =

fmmd--hybrid3$(EXEEXT): \
      $(fmmd_hybrid3_OBJS) \
      $(fmmd_hybrid3_CUOBJS) \
      $(fmmd_hybrid3_CXXOBJS) \
      $(fmmd_hybrid3_LIBS) \
      $(fmmd_common_OBJS) \
      $(fmmd_common_LIBS) \
      $(fmm_common_LIBS)
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) -DUSE_NVCC ${INCLUDES} -o $@ $^ \
      $(OMPFLAGS) $(CXXLDFLAGS) ${LDFLAGS} 

CLEANFILES += $(fmmd_hybrid3_CUOBJS)
CLEANFILES += $(fmmd_hybrid3_CXXOBJS)
#=======================================================================


#=======================================================================
CLEANFILES += $(fmm_common_OBJS) $(fmm_common_DEPS)
CLEANFILES += $(fmm_naive_OBJS) $(fmm_naive_DEPS)

#=======================================================================

%.o__d: %.c
	$(CC) $(CFLAGS_STD) $(CFLAGS) $(COPTFLAGS) $(OMPFLAGS) \
	    -o $@ -c $<

%.o__s: %.c
	$(CC) $(CFLAGS_STD) $(CFLAGS) $(COPTFLAGS) $(OMPFLAGS) \
	    -DUSE_FLOAT -o $@ -c $<

%.o__d: %.cc
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) $(OMPFLAGS) \
	    -o $@ -c $<

%.o__s: %.cc
	$(CXX) $(CXXFLAGS_STD) $(CXXFLAGS) $(CXXOPTFLAGS) $(OMPFLAGS) \
	    -DUSE_FLOAT -o $@ -c $<

%.o__d: %.cu
	$(NVCC) -g -Xcompiler -fopenmp -DUSE_NVCC $(INCLUDES) -o $@ -c $< 

#=======================================================================
# eof
