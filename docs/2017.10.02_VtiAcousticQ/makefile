
all:	2017.10.02_VtiAcousticQ.pdf

2017.10.02_VtiAcousticQ.pdf: 2017.10.02_VtiAcousticQ.tex
	pdflatex 2017.10.02_VtiAcousticQ.tex
	pdflatex 2017.10.02_VtiAcousticQ.tex

clean:
	/bin/rm -rf *.aux *.log *.pdf

view: 2017.10.02_VtiAcousticQ.pdf
	gv -watch 2017.10.02_VtiAcousticQ.pdf &
