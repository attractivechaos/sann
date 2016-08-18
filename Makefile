CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -Wno-unused-function -O2
CPPFLAGS=	#-D_NO_SSE
INCLUDES=	-I.
OBJS=		math.o sae.o smln.o sann.o data.o
PROG=		sann
LIBS=		-lm -lz -lpthread

.SUFFIXES:.c .o

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:libsann.a $(PROG)

sann:cli.o cli_priv.o libsann.a
		$(CC) $(CFLAGS) cli.o cli_priv.o -o $@ -L. -lsann $(LIBS)

libsann.a:$(OBJS)
		$(AR) -csru $@ $(OBJS)

clean:
		rm -fr gmon.out *.o a.out $(PROG) $(PROG_EXTRA) *~ *.a *.dSYM session*

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c)

# DO NOT DELETE

cli.o: sann.h
cli_priv.o: priv.h sann.h
data.o: sann.h priv.h kseq.h
example.o: sann.h
math.o: sann.h priv.h
sae.o: priv.h sann.h ksort.h
sann.o: priv.h sann.h
smln.o: priv.h sann.h
