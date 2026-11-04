# Build the architecture figure in the project's LaTeX container.
DOCKER_IMAGE = oaklight/texlive:science-cn
DOCKER_RUN   = docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
               -v "$(CURDIR)":/workspace -w /workspace $(DOCKER_IMAGE)

SRC = architecture.tex
PDF = architecture.pdf

.PHONY: all build clean clean-aux fmt

all: build

build: $(PDF)

$(PDF): $(SRC)
	$(DOCKER_RUN) pdflatex -interaction=nonstopmode -halt-on-error $(SRC)
	$(DOCKER_RUN) pdflatex -interaction=nonstopmode -halt-on-error $(SRC)

clean:
	rm -f $(PDF) *.aux *.log *.out

clean-aux:
	rm -f *.aux *.log *.out

fmt:
	$(DOCKER_RUN) latexindent -w -s $(SRC)
