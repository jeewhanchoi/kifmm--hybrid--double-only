top_srcdir = ..
include $(top_srcdir)/Makefile.inc

#=======================================================================

TARGETS = libtiming.a
CLEANFILES = $(TARGETS)

all: $(TARGETS)
#=======================================================================

libtiming_a_HDRS = timing.h
libtiming_a_SRCS = $(libtiming_a_HDRS:.h=.c)
libtiming_a_OBJS = $(libtiming_a_SRCS:.c=.o)
libtiming_a_DEPS = $(libtiming_a_SRCS:.c=.d)
-include $(libtiming_a_DEPS)

libtiming.a: $(libtiming_a_OBJS)
	$(AR) $(ARCRFLAGS) $@ $(libtiming_a_OBJS)
	$(RANLIB) $@

CLEANFILES += $(libtiming_a_OBJS)

#=======================================================================
enum_papi_events$(EXEEXT): enum_papi_events.c
	$(CC) $(CFLAGS) -o $@ $< $(CLDFLAGS)

CLEANFILES += enum_papi_events$(EXEEXT)

# eof
