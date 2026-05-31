CC=gcc
VPATH=src tests include
CPPFLAGS=-I include
MARCH ?=
CFLAGS=-O3 -fPIC -std=c99 -ffast-math -fno-cx-limited-range $(MARCH) $(OPENMP_CFLAGS)
#CFLAGS=-g -O -fPIC -I$(LAPACKE_INCLUDE)
#CFLAGS=-g -O -fPIC -I$(LAPACKE_INCLUDE)
LDLIBS=-llapacke -llapack -lblas -lgfortran -lgslcblas -lgsl -lm -lnlopt
LDFLAGS=$(OPENMP_LDFLAGS)
# NOTE: -openmp is the legacy icc flag; deprecated in icc 16.0 and removed in
# Intel oneAPI (icx). If building with a modern Intel compiler, change to
# -qopenmp (and likewise in setup.py's icc branch).
MKL_CFLAGS=-mkl -O3 -fPIC -std=c99 -ansi-alias -march=x86-64 -mtune=generic -lpthread -openmp
MKL_LDLIBS=-lmkl_intel_lp64 -lgsl -lnlopt
MKL_LDFLAGS=-Wl,-rpath,$(INTEL_PATH)/lib/intel64/ -Wl,-rpath,$(INTEL_PATH)/mkl/lib/intel64/
ARFLAGS=rcs

# OpenMP flags for the GCC build path. GCC on Linux ships libgomp; override
# OPENMP_CFLAGS / OPENMP_LDFLAGS on platforms without it (e.g. stock Apple
# clang -- use Homebrew libomp). To disable entirely:
#   make OPENMP_CFLAGS= OPENMP_LDFLAGS=
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  OPENMP_CFLAGS ?= -fopenmp
  OPENMP_LDFLAGS ?= -fopenmp
else
  OPENMP_CFLAGS ?=
  OPENMP_LDFLAGS ?=
endif

ifeq ($(CFL_CC), icc)
  CC=icc
  CFLAGS=$(MKL_CFLAGS)
  LDLIBS=$(MKL_LDLIBS)
  LDFLAGS=$(MKL_LDFLAGS)
endif

ifdef CFL_CFLAGS
  CFLAGS+=$(CFL_CFLAGS)
else
  # Fallback when CFL_CFLAGS is not provided (e.g., manual make invocation)
  # Support both /usr/include/lapacke.h and /usr/include/lapacke/lapacke.h
  CFLAGS+=-I/usr/include -I/usr/include/lapacke
endif
ifdef CFL_LDLIBS
  LDLIBS+=$(CFL_LDLIBS)
endif

TESTS=$(patsubst tests/%.c,%,$(wildcard tests/*test.c))

all: libcfl.a

.PHONY: all test debug asan clean

test: $(TESTS)
	sh ./cfl_testing.sh

debug: CFLAGS=-g -O1 -fPIC -std=c99 -I/usr/include/lapacke $(OPENMP_CFLAGS)
debug: all

# AddressSanitizer + UndefinedBehaviorSanitizer build.
# Usage:  make -C cfl asan
# Then run individual test binaries directly, e.g. ./cfl/h_test.
# OpenMP is disabled in this build because libgomp interacts poorly with
# AddressSanitizer (spurious leak reports from the runtime's worker pools).
asan: CFLAGS=-g -O2 -fPIC -std=c99 -fno-omit-frame-pointer -fsanitize=address,undefined -I/usr/include -I/usr/include/lapacke
asan: LDFLAGS=-fsanitize=address,undefined
asan: OPENMP_CFLAGS=
asan: OPENMP_LDFLAGS=
asan: clean libcfl.a $(TESTS)

libcfl.a: cfl_error.o cfl_zefoz.o cfl_h_fit.o basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_csr.o
	$(AR) $(ARFLAGS) $@ $^

zefoz_test: cfl_error.o cfl_zefoz.o cfl_h.o cfl_tensor.o cfl_csr.o

siman_test: cfl_error.o cfl_h_fit.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_csr.o

nls_test: cfl_error.o cfl_h_fit.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_csr.o

opt_test: cfl_error.o cfl_h_fit.o basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_csr.o

sh_test: cfl_error.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_csr.o

sh_zeeman_test: cfl_error.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_csr.o

h_test: cfl_error.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_csr.o

csr_test: cfl_error.o cfl_tensor.o cfl_h.o cfl_csr.o

cfl_error.o: cfl_error.h

cfl_zefoz.o: cfl_h.o cfl_zefoz.h cfl_error.h cfl_config.h

cfl_h_fit.o: basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_h_fit.h cfl_error.h

basinhopping.o: basinhopping.h cfl_error.h cfl_min.o cfl_config.h

cfl_min.o: cfl_min.h cfl_error.h cfl_config.h

cfl_sh.o: cfl_tensor.o cfl_sh.h cfl_error.h

cfl_h.o: cfl_tensor.o cfl_h.h cfl_error.h cfl_config.h

cfl_tensor.o: cfl_csr.o cfl_tensor.h cfl_error.h

cfl_csr.o: cfl_csr.h cfl_error.h

clean:
	rm -f *.o
	rm -f $(TESTS) test
