VPATH=src include
CPPFLAGS=-I include
CFLAGS=-g -O1 -fPIC 
#CFLAGS=-O3 -fprofile-generate -fPIC -mtune=native
#CFLAGS=-O3 -fprofile-use -fPIC -mtune=native
LDLIBS=-llapacke -lblas -lm -lgsl
LDFLAGS=

#libcfl.so: h_fit.o basinhopping.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o 

#cfl_test: cfl_h.o cfl_tensor.o cfl_crs.o -lm

opt_test: h_fit.o basinhopping.o cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o test_data.h cfl_error.h -lm 
#h_test: cfl_sh.o cfl_h.o cfl_tensor.o cfl_crs.o test_data.h -lm
#crs_test: cfl_h.o cfl_crs.o -lm
#libcfl.a: cfl_h.o cfl_crs.o 
#	ar -rs $@ $^

h_fit.o: basinhopping.o cfl_sh.o cfl_h.o h_fit.h cfl_error.h

basinhopping.o: basinhopping.h cfl_error.h

cfl_sh.o: cfl_tensor.o cfl_sh.h cfl_error.h

cfl_h.o: cfl_tensor.o cfl_h.h cfl_error.h

cfl_tensor.o: cfl_crs.o cfl_tensor.h cfl_error.h

cfl_crs.o: cfl_crs.h cfl_error.h

clean:
	rm -f *.o
	rm -f *.gcda
