# Build the readout-taxonomy figure in the project's LaTeX container.
DOCKER_IMAGE = oaklight/texlive:science-cn
DOCKER_RUN   = docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
               -v "$(CURDIR)":/workspace -w /workspace $(DOCKER_IMAGE)

SRC = readout-taxonomy.tex
PDF = readout-taxonomy.pdf

.PHONY: all build clean clean-aux png

all: build

build: $(PDF)

$(PDF): $(SRC)
	$(DOCKER_RUN) pdflatex -interaction=nonstopmode -halt-on-error $(SRC)
	$(DOCKER_RUN) pdflatex -interaction=nonstopmode -halt-on-error $(SRC)

png: $(PDF)
	pdftoppm -png -r 150 $(PDF) preview

clean:
	rm -f $(PDF) preview-*.png *.aux *.log *.out

clean-aux:
	rm -f *.aux *.log *.out
