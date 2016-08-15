CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -Wno-unused-function -O2
CPPFLAGS=
FANN_PREFIX=$(HOME)/fann
INCLUDES=	-I.. -I$(FANN_PREFIX)/include
PROG=		sann-fann
LIBS=		-L$(FANN_PREFIX)/lib -lfann -lm -lz

.SUFFIXES:.c .o

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(PROG)

sann-fann:sann-fann.o ../data.o
		$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
		rm -fr gmon.out *.o a.out $(PROG) $(PROG_EXTRA) *~ *.a *.dSYM session*

sann-fann.o: ../sann.h
