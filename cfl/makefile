CC=gcc
VPATH=src tests include
CPPFLAGS=-I include
CFLAGS=-g -O3 -fPIC -std=c99 -ffast-math -fno-cx-limited-range -march=native -I/usr/include/lapacke 
CFLAGS=-g -O1 -fPIC -std=c99 -I/usr/include/lapacke 
LDLIBS=-llapacke -llapack -lblas -lgfortran -lgslcblas -lgsl -lm -lnlopt 
LDFLAGS=
MKL_CFLAGS=-mkl -g -O3 -fPIC -std=c99 -ansi-alias -xhost -lpthread -openmp
MKL_LDLIBS=-lmkl_intel_lp64 -lgsl -lnlopt
MKL_LDFLAGS=-Wl,-rpath,$(INTEL_PATH)/lib/intel64/ -Wl,-rpath,$(INTEL_PATH)/mkl/lib/intel64/
ARFLAGS=rcs

ifeq ($(CFL_CC), icc)
  CC=icc
  CFLAGS=$(MKL_CFLAGS)
  LDLIBS=$(MKL_LDLIBS)
  LDFLAGS=$(MKL_LDFLAGS)
endif  

ifdef CFL_CFLAGS
  CFLAGS+=$(CFL_CFLAGS)
endif
ifdef CFL_LDLIBS
  LDLIBS+=$(CFL_LDLIBS) 
endif

TESTS=$(patsubst tests/%.c,%,$(wildcard tests/*test.c))

all: libcfl.a

test: $(TESTS) 
	sh ./cfl_testing.sh
	touch test

debug: CFLAGS=-g -O1 -fPIC -std=c99 -I/usr/include/lapacke 
debug: all

libcfl.a: cfl_h_fit.o basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o rcm.o cfl_error.h 
	$(AR) $(ARFLAGS) $@ $^

opt_test: cfl_h_fit.o basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o cfl_error.h 

sh_test: cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o 

sh_zeeman_test: cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o 

h_test: cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o rcm.o

crs_test: cfl_h.o cfl_crs.o rcm.o

cfl_h_fit.o: basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_h_fit.h cfl_error.h

basinhopping.o: basinhopping.h cfl_error.h cfl_min.o cfl_config.h

cfl_min.o: cfl_min.h cfl_error.h cfl_config.h

cfl_sh.o: cfl_tensor.o cfl_sh.h cfl_error.h

cfl_h.o: rcm.o cfl_tensor.o cfl_h.h cfl_error.h cfl_config.h

cfl_tensor.o: cfl_crs.o cfl_tensor.h cfl_error.h

cfl_crs.o: cfl_crs.h cfl_error.h

rcm.o: rcm.h

clean:
	rm -f *.o
	rm -f $(TESTS) test



