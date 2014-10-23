CC=gcc
VPATH=src tests include
CPPFLAGS=-I include
CFLAGS=-g -O3 -fPIC -std=c99 -ffast-math -fno-cx-limited-range -march=native -fprofile-use
LDLIBS=-llapacke -lblas -lm -lgsl -lnlopt
MKL_CFLAGS=-g -O3 -fPIC -std=c99 -mkl -ansi-alias -xhost -parallel -prof-use 
MKL_LDLIBS=-lmkl_intel_lp64 -lgsl -lnlopt
LDFLAGS=
ARFLAGS=rcs

TESTS=$(patsubst tests/%.c,%,$(wildcard tests/*test.c))

all: libcfl.a test

test: $(TESTS) 
	sh ./cfl_testing.sh

mkl: CC=icc
mkl: CFLAGS=$(MKL_CFLAGS)
mkl: LDLIBS=$(MKL_LDLIBS)
mkl: all

debug: CFLAGS=-g -O1 -fPIC 
debug: all

libcfl.a: cfl_h_fit.o basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o cfl_error.h -lm
	$(AR) $(ARFLAGS) $@ $^

opt_test: cfl_h_fit.o basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o cfl_error.h -lm

h_test: cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o -lm

crs_test: cfl_h.o cfl_crs.o -lm

cfl_h_fit.o: basinhopping.o cfl_min.o cfl_sh.o cfl_h.o cfl_h_fit.h cfl_error.h

basinhopping.o: basinhopping.h cfl_error.h cfl_min.o

cfl_min.o: cfl_min.h cfl_error.h

cfl_sh.o: cfl_tensor.o cfl_sh.h cfl_error.h

cfl_h.o: cfl_tensor.o cfl_h.h cfl_error.h cfl_config.h

cfl_tensor.o: cfl_crs.o cfl_tensor.h cfl_error.h

cfl_crs.o: cfl_crs.h cfl_error.h

clean:
	rm -f *.o
	rm -f $(TESTS)



