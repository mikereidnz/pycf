VPATH=src include
CPPFLAGS=-I include
CFLAGS=-g -O1 -fPIC 
LDLIBS=-llapacke -lblas -lm -lgsl
LDFLAGS=

#libcfl.so: cfl_h.o cfl_crs.o 

#cfl_test: cfl_h.o cfl_tensor.o cfl_crs.o -lm

opt_test: cfl_error.h basinhopping.o -lm
#h_test: cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o -lm
#crs_test: cfl_h.o cfl_crs.o -lm
#libcfl.a: cfl_h.o cfl_crs.o 
#	ar -rs $@ $^

cfl_h.o: cfl_tensor.o cfl_h.h cfl_error.h

cfl_tensor.o: cfl_crs.o cfl_tensor.h cfl_error.h

cfl_crs.o: cfl_crs.h cfl_error.h

clean:
	rm -f *.o

