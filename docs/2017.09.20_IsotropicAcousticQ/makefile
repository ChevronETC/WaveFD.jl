
all:	2017.09.20_IsotropicAcousticQ.pdf

2017.09.20_IsotropicAcousticQ.pdf: 2017.09.20_IsotropicAcousticQ.tex
	pdflatex 2017.09.20_IsotropicAcousticQ.tex
	pdflatex 2017.09.20_IsotropicAcousticQ.tex

clean:
	/bin/rm -rf 2017.09.20_IsotropicAcousticQ.aux 2017.09.20_IsotropicAcousticQ.log 2017.09.20_IsotropicAcousticQ.pdf

view: 2017.09.20_IsotropicAcousticQ.pdf
	gv -watch ./2017.09.20_IsotropicAcousticQ.pdf &
