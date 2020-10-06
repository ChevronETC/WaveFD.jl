
all:	IsotropicAcousticQ.pdf VtiAcousticQ.pdf

IsotropicAcousticQ.pdf: IsotropicAcousticQ.tex
	pdflatex IsotropicAcousticQ.tex
	pdflatex IsotropicAcousticQ.tex

VtiAcousticQ.pdf: VtiAcousticQ.tex
	pdflatex VtiAcousticQ.tex
	pdflatex VtiAcousticQ.tex

clean:
	/bin/rm -rf *.aux *.log *.pdf
