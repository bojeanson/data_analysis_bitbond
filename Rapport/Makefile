all: compile clean

compile:
	pdflatex -shell-escape rapport.tex
	pdflatex -shell-escape rapport.tex
	bibtex rapport
	pdflatex -shell-escape rapport.tex
	pdflatex -shell-escape rapport.tex
	
clean:
	rm *.aux 2>/dev/null; true
	rm *.log 2>/dev/null; true
	rm *.out 2>/dev/null; true
	rm *.bbl 2>/dev/null; true
	rm *.blg 2>/dev/null; true
	rm *.toc 2>/dev/null; true
	rm *.run.xml 2>/dev/null; true
	rm *blx.bib 2>/dev/null; true
	rm *.bcf 2>/dev/null; true
	rm parties/*.aux 2>/dev/null; true
