VPATH=src include
CPPFLAGS=-I include
CFLAGS=-g -O1 -fPIC 
LDLIBS=-llapacke -lm
LDFLAGS=

#libcfl.so: cfl_h.o cfl_crs.o 

#cfl_test: cfl_h.o cfl_crs.o -lm
libcfl.a: cfl_h.o cfl_crs.o 
	ar -rs $@ $^

cfl_h.o: cfl_crs.o cfl_h.h cfl_error.h

cfl_crs.o: cfl_crs.h cfl_error.h

clean:
	rm -f *.o

