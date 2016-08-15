CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -Wno-unused-function -O2
CPPFLAGS=
INCLUDES=	-I.
OBJS=
PROG=		mnist2snd
LIBS=		-lm -lz

.SUFFIXES:.c .o

.c.o:
		$(CC) -c $(CFLAGS) -I.. $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(PROG)

mnist2snd:mnist.o mnist2snd.o
		$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
		rm -fr gmon.out *.o a.out $(PROG) $(PROG_EXTRA) *~ *.a *.dSYM session*

mnist.o: mnist.h ../kseq.h
mnist2snd.o: mnist.h
