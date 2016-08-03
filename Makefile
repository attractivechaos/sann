CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -Wno-unused-function -O2
CPPFLAGS=
INCLUDES=	-I.
OBJS=		math.o sae_core.o smln_core.o sann.o data.o
PROG=		sann
LIBS=		-lm -lz -lpthread

.SUFFIXES:.c .o

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(PROG)

sann:$(OBJS) cli.o
		$(CC) $(CFLAGS) $^  -o $@ $(LIBS)

clean:
		rm -fr gmon.out *.o a.out $(PROG) $(PROG_EXTRA) *~ *.a *.dSYM session*

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c)

# DO NOT DELETE

cli.o: sann.h
data.o: sann.h kseq.h
math.o: sann.h priv.h
sae_core.o: sann.h priv.h ksort.h
sae_misc.o: sann.h
sann.o: priv.h sann.h
smln_core.o: sann.h priv.h
