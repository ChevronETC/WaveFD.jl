#For GNU --
CC      := gcc -std=c99
CXX     := g++ -std=c++11
CFLAGS  := -fPIC -fopenmp -Ofast -ftree-vectorize -mavx2 -mavx -m64 -fstrict-aliasing -funroll-loops
LDFLAGS := -shared -fopenmp

all : \
	libprop2DAcoIsoDenQ_DEO2_FDTD.so \
	libprop2DAcoTTIDenQ_DEO2_FDTD.so \
	libprop2DAcoVTIDenQ_DEO2_FDTD.so \
	libprop3DAcoIsoDenQ_DEO2_FDTD.so \
	libprop3DAcoTTIDenQ_DEO2_FDTD.so \
	libprop3DAcoVTIDenQ_DEO2_FDTD.so \
	libillumination.so \
	libspacetime.so

# 2D Isotropic acoustic, variable density, amplitude only Q absorbing boundaries, coupled second order
libprop2DAcoIsoDenQ_DEO2_FDTD.so: prop2DAcoIsoDenQ_DEO2_FDTD.cc prop2DAcoIsoDenQ_DEO2_FDTD.h absorb.h
	$(CXX) ${CFLAGS} -I. -c prop2DAcoIsoDenQ_DEO2_FDTD.cc
	$(CC) ${LDFLAGS} -o libprop2DAcoIsoDenQ_DEO2_FDTD.so prop2DAcoIsoDenQ_DEO2_FDTD.o

# 2D TTI pseudo-acoustic, variable density, amplitude only Q absorbing boundaries, coupled second order
libprop2DAcoTTIDenQ_DEO2_FDTD.so: prop2DAcoTTIDenQ_DEO2_FDTD.cc prop2DAcoTTIDenQ_DEO2_FDTD.h absorb.h
		$(CXX) ${CFLAGS} -I. -c prop2DAcoTTIDenQ_DEO2_FDTD.cc
		$(CC) ${LDFLAGS} -o libprop2DAcoTTIDenQ_DEO2_FDTD.so prop2DAcoTTIDenQ_DEO2_FDTD.o

# 2D VTI pseudo-acoustic, variable density, amplitude only Q absorbing boundaries, coupled second order
libprop2DAcoVTIDenQ_DEO2_FDTD.so: prop2DAcoVTIDenQ_DEO2_FDTD.cc prop2DAcoVTIDenQ_DEO2_FDTD.h absorb.h
	$(CXX) ${CFLAGS} -I. -c prop2DAcoVTIDenQ_DEO2_FDTD.cc
	$(CC) ${LDFLAGS} -o libprop2DAcoVTIDenQ_DEO2_FDTD.so prop2DAcoVTIDenQ_DEO2_FDTD.o

# 3D Isotropic acoustic, variable density, amplitude only Q absorbing boundaries, coupled second order
libprop3DAcoIsoDenQ_DEO2_FDTD.so: prop3DAcoIsoDenQ_DEO2_FDTD.cc prop3DAcoIsoDenQ_DEO2_FDTD.h absorb.h
	$(CXX) ${CFLAGS} -I. -c prop3DAcoIsoDenQ_DEO2_FDTD.cc
	$(CC) ${LDFLAGS} -o libprop3DAcoIsoDenQ_DEO2_FDTD.so prop3DAcoIsoDenQ_DEO2_FDTD.o

# 3D TTI pseudo-acoustic, variable density, amplitude only Q absorbing boundaries, coupled second order
libprop3DAcoTTIDenQ_DEO2_FDTD.so: prop3DAcoTTIDenQ_DEO2_FDTD.cc prop3DAcoTTIDenQ_DEO2_FDTD.h absorb.h
	$(CXX) ${CFLAGS} -I. -c prop3DAcoTTIDenQ_DEO2_FDTD.cc
	$(CC) ${LDFLAGS} -o libprop3DAcoTTIDenQ_DEO2_FDTD.so prop3DAcoTTIDenQ_DEO2_FDTD.o

# 3D VTI pseudo-acoustic, variable density, amplitude only Q absorbing boundaries, coupled second order
libprop3DAcoVTIDenQ_DEO2_FDTD.so: prop3DAcoVTIDenQ_DEO2_FDTD.cc prop3DAcoVTIDenQ_DEO2_FDTD.h absorb.h
	$(CXX) ${CFLAGS} -I. -c prop3DAcoVTIDenQ_DEO2_FDTD.cc
	$(CC) ${LDFLAGS} -o libprop3DAcoVTIDenQ_DEO2_FDTD.so prop3DAcoVTIDenQ_DEO2_FDTD.o

# .................................................................................................
# Auxiliary libraries
# .................................................................................................
libspacetime.so: spacetime.cc
	${CXX} ${CFLAGS} -c spacetime.cc
	${CC} ${LDFLAGS} -o libspacetime.so spacetime.o

libillumination.so: illumination.c
	${CC} ${CFLAGS} -c illumination.c
	${CC} ${LDFLAGS} -o libillumination.so illumination.o

clean:
	rm -f lib*.so prop*.o *.optrpt *.output *.o

install: all
	mkdir -p ../deps/usr/lib
	cp absorb.h ../deps/usr/include
	cp libillumination.so ../deps/usr/lib
	cp libspacetime.so ../deps/usr/lib
	cp libprop2DAcoIsoDenQ_DEO2_FDTD.so ../deps/usr/lib
	cp libprop2DAcoTTIDenQ_DEO2_FDTD.so ../deps/usr/lib
	cp libprop2DAcoVTIDenQ_DEO2_FDTD.so ../deps/usr/lib
	cp libprop3DAcoIsoDenQ_DEO2_FDTD.so ../deps/usr/lib
	cp libprop3DAcoTTIDenQ_DEO2_FDTD.so ../deps/usr/lib
	cp libprop3DAcoVTIDenQ_DEO2_FDTD.so ../deps/usr/lib
	cp prop2DAcoIsoDenQ_DEO2_FDTD*.h ../deps/usr/include
	cp prop2DAcoTTIDenQ_DEO2_FDTD*.h ../deps/usr/include
	cp prop2DAcoVTIDenQ_DEO2_FDTD.h ../deps/usr/include
	cp prop3DAcoIsoDenQ_DEO2_FDTD.h ../deps/usr/include
	cp prop3DAcoTTIDenQ_DEO2_FDTD.h ../deps/usr/include
	cp prop3DAcoVTIDenQ_DEO2_FDTD.h ../deps/usr/include