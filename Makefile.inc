all-targets: announce all-recursive all-local

#=======================================================================
# Auto-detect an appropriate configuration.
#
# Override by setting environment variable, 'BUILD_ARCH'.
#=======================================================================

ifeq ($(host-name),)
  host-name := $(shell hostname)
endif

ifeq ($(kernel-name),)
  kernel-name := $(shell uname -s)
endif

ifeq ($(arch-name),)
  arch-name := $(shell uname -p)
endif

ifeq ($(proc-name),)

  # Try to detect processor

  ifeq ($(detect-cpuinfo),)
    ifeq ($(kernel-name),Darwin)
      detect-cpuinfo := $(shell sysctl machdep.cpu.brand_string | grep "^.*Xeon")
    endif
    ifeq ($(kernel-name),Linux)
      detect-cpuinfo := $(shell grep "^model name.*Xeon" /proc/cpuinfo)
    endif
    ifneq ($(detect-cpuinfo),)
      proc-name := Xeon
      proc-vendor := Intel
    endif
  endif

  ifeq ($(detect-cpuinfo),)
    ifeq ($(kernel-name),Darwin)
      detect-cpuinfo := $(shell sysctl machdep.cpu.brand_string | grep "^.*Core(TM)2 Duo")
    endif
    ifeq ($(kernel-name),Linux)
      detect-cpuinfo := $(shell grep "^model name.*Core(TM)2 Duo" /proc/cpuinfo)
    endif
    ifneq ($(detect-cpuinfo),)
      proc-name := Core2Duo
      proc-vendor := Intel
    endif
  endif

  ifeq ($(detect-cpuinfo),)
    detect-cpuinfo := $(shell grep "^model name.*Quad-Core .*Opteron" /proc/cpuinfo)
    ifneq ($(detect-cpuinfo),)
      proc-name := Barcelona
      proc-vendor := AMD
    endif
  endif

  ifeq ($(proc-name),)
    proc-name := unknown
    proc-vendor := unknown
  endif
endif

ifeq ($(cc-name),)
# GNU compiler check DISABLED
#  cc-name := $(shell basename $(shell which gcc))
  ifeq ($(cc-name),)
    cc-name := $(shell basename $(shell which icc))
  endif
endif

ifeq ($(BUILD_ARCH),)
  BUILD_ARCH = $(arch-name)-$(proc-name)-$(kernel-name)-$(cc-name)
endif

include $(top_srcdir)/arch/$(BUILD_ARCH)

#=======================================================================
# Default build rules
#=======================================================================

CFLAGS_STD =
CFLAGS_STD += -I.
CFLAGS_STD += -I$(top_srcdir)/timing
CFLAGS_STD += -DPROCESSOR=\"$(proc-name)\"
CFLAGS_STD += -DKERNEL=\"$(kernel-name)\"

CXXFLAGS_STD = $(CFLAGS_STD)

%.o: %.c
	$(CC) $(CFLAGS_STD) $(CDEPFLAGS) $(CFLAGS) $(OMPFLAGS) $(COPTFLAGS) -o $@ -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS_STD) $(CXXDEPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(CXXOPTFLAGS) -o $@ -c $<

%.o: %.f
	$(FC) $(FFLAGS) $(FOPTFLAGS) -o $@ -c $<

announce: hline-top announce-arch hline-bottom

hline-top:
	@echo "=================================================="

hline-bottom:
	@echo "=================================================="

announce-arch:
	@echo "Build architecture: $(BUILD_ARCH)"

all-local: all

all-recursive:
	@$(MAKE) recursive RECTGT=all

clean: clean-local clean-recursive

clean-recursive:
	@$(MAKE) recursive RECTGT=clean

clean-local:
	rm -rf core *~ *.d $(TARGETS) $(CLEANFILES)

recursive:
	@test -n "$(RECTGT)"
	@for subdir in $(SUBDIRS) ; do \
	  echo "=== $(RECTGT): Entering subdirectory, `pwd`/$$subdir ... ===" ; \
	  cd $$subdir || exit 1 ; \
	  $(MAKE) $(RECTGT) || exit 1 ; \
	  cd - ; \
	done

# eof
