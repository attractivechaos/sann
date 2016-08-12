.SUFFIXES: .gp .tex .eps .pdf .eps.gz

.eps.pdf:
		epstopdf --outfile $@ $<

.eps.gz.pdf:
		gzip -dc $< | epstopdf --filter > $@

.pdf.eps:
		pdftops -eps $< $@

all:dann.pdf

dann.pdf:dann.tex
		pdflatex $<; pdflatex $<;

clean:
		rm -fr *.toc *.aux *.bbl *.blg *.idx *.log *.out *~ dann.pdf
