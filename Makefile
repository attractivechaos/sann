CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
ZLIB_FLAGS=	-DHAVE_ZLIB   # comment out this line to drop the zlib dependency
INCLUDES=	-I.
OBJS=		math.o sae.o sfnn.o sann.o data.o io.o
PROG=		sann
LIBS=		-lm -lz

.SUFFIXES:.c .o
.PHONY:all demo clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:libsann.a $(PROG)

demo:xor-demo sann-demo

sann:cli.o cli_priv.o libsann.a
		$(CC) $(CFLAGS) cli.o cli_priv.o -o $@ -L. -lsann $(LIBS)

libsann.a:$(OBJS)
		$(AR) -csru $@ $(OBJS)

data.o:data.c
		$(CC) -c $(CFLAGS) $(ZLIB_FLAGS) $(INCLUDES) $< -o $@

sann-demo:demo.c libsann.a
		$(CC) $(CFLAGS) $< -o $@ -L. -lsann $(LIBS)

xor-demo:xor-demo.c libsann.a
		$(CC) $(CFLAGS) $< -o $@ -L. -lsann $(LIBS)

clean:
		rm -fr gmon.out *.o a.out $(PROG) $(PROG_EXTRA) *~ *.a *.dSYM session* xor-demo sann-demo

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c)

# DO NOT DELETE

cli.o: sann.h
cli_priv.o: sann_priv.h sann.h
data.o: sann.h kseq.h
demo.o: sann.h
io.o: sann.h
math.o: sann.h sann_priv.h
sae.o: sann_priv.h sann.h
sann.o: sann_priv.h sann.h
sfnn.o: sann_priv.h sann.h
xor-demo.o: sann.h
